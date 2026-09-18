#!/usr/bin/env Rscript
# Build the depth screen's arms with splatter instead of by hand.
#
# The screen's own construction draws lognormal gene means, adds gamma
# overdispersion, and applies depth by multinomial resampling. That makes depth
# exactly orthogonal to expression, which is the cleanest possible null, but the
# simulator is ours. Splatter is published and has this knob already:
# `lib.loc` and `lib.scale` draw each cell's library size from
# rlnorm(n, lib.loc, lib.scale), the same lognormal parameterisation, so the
# depth ladder can be expressed in splatter's own parameters -- lib.loc =
# log(median depth), lib.scale = the sigma the arm is named for.
#
# One consequence to be aware of rather than hide. Splatter applies its
# biological coefficient of variation *after* library-size scaling, and the BCV
# falls with the mean, so a deep cell sits closer to its group centroid than a
# shallow one. Depth therefore does shape the geometry here, through a
# mechanism that is real -- it is the depth-graded dispersion measured on the
# actual data, rho(depth, nearest-neighbour distance) = -0.41. The correct
# answer is still "reject nothing", because no cell comes from another
# population, so this remains a valid specificity test. It is a harder and more
# realistic one, and it is not the exactly-orthogonal null. Both are worth
# having, which is why this sits beside the hand-built path rather than
# replacing it.
#
# Arms are composed the way the scaling benchmark composes its scenarios: one
# pool, two batches with no batch effect, source drawn from the first and target
# from the second, so the two sides are independent draws from one population.
# The perturbed arm takes its extra 20% from a second group. Every arm comes
# from the same simulator call with the same parameters, so the arms differ only
# in composition and in lib.scale.
#
# The perturbation is set to reproduce the hand-built one inside splatter's own
# machinery: 10% of genes at 2x, upward only. de.facLoc = log(2) with a narrow
# scale centres the multiplier on 2, and de.downProb = 0 stops splatter sending
# half of them down.
#
# Written to fail loudly. Splatter's lib.scale is an assumption about somebody
# else's package, so the realised sd of log library size is measured for every
# arm and compared against what was asked for; a mismatch beyond tolerance
# aborts instead of quietly producing a ladder with the wrong rungs.
#
# Usage:
#   Rscript scripts/generate_splatter_depth_arms.R OUTPUT_DIR \
#       [n_cells] [n_genes] [replicates] [seed] [median_depth] \
#       [sigmas] [perturbed_fraction] [tolerance]
#
# Smoke test first, because this cannot be run without splatter present:
#   Rscript scripts/generate_splatter_depth_arms.R /tmp/smoke 120 400 1

suppressPackageStartupMessages({
  library(Matrix)
  library(splatter)
  library(SingleCellExperiment)
  library(SummarizedExperiment)
})

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) < 1L) {
  stop("usage: generate_splatter_depth_arms.R OUTPUT_DIR [n_cells] [n_genes] ",
       "[replicates] [seed] [median_depth] [sigmas] [perturbed_fraction] ",
       "[tolerance]")
}

pick <- function(index, fallback) {
  if (length(arguments) >= index && nzchar(arguments[[index]])) {
    arguments[[index]]
  } else {
    fallback
  }
}

output_root <- arguments[[1]]
n_cells <- as.integer(pick(2L, "1500"))
n_genes <- as.integer(pick(3L, "4000"))
replicates <- as.integer(pick(4L, "3"))
base_seed <- as.integer(pick(5L, "20260914"))
median_depth <- as.numeric(pick(6L, "10000"))
sigmas <- as.numeric(strsplit(pick(7L, "0,0.3,0.6,0.9"), ",", fixed = TRUE)[[1]])
perturbed_fraction <- as.numeric(pick(8L, "0.2"))
tolerance <- as.numeric(pick(9L, "0.08"))
# Splatter holds several dense cells-by-genes matrices at once, so the pool is
# sized to what the composition needs rather than generously. Per batch the
# most that is drawn is n cells of group 1, and group 1 is 70% of the pool, so
# 2n per batch leaves a 40% margin; take() aborts if a draw ever falls short.
pool_multiplier <- as.integer(pick(10L, "2"))

stopifnot(
  n_cells >= 20L, n_genes >= 50L, replicates >= 1L,
  median_depth > 0, all(is.finite(sigmas)), all(sigmas >= 0),
  perturbed_fraction > 0, perturbed_fraction < 1, tolerance > 0,
  pool_multiplier >= 2L
)

# The names the Python runner already uses, so nothing downstream needs new
# arm labels. A sigma outside the ladder gets a descriptive name instead.
homogeneous_name <- function(sigma) {
  if (isTRUE(all.equal(sigma, 0.0))) return("homogeneous_depth_cv0")
  if (isTRUE(all.equal(sigma, 0.3))) return("homogeneous_depth_cv_low")
  if (isTRUE(all.equal(sigma, 0.6))) return("homogeneous_depth_cv_mid")
  if (isTRUE(all.equal(sigma, 0.9))) return("homogeneous_depth_cv_high")
  sprintf("homogeneous_depth_sigma_%s", sub(".", "p", format(sigma), fixed = TRUE))
}

# group 1 is the shared population, group 2 the source-only perturbation.
GROUP_PROBABILITY <- c(0.7, 0.3)

simulate_pool <- function(sigma, seed) {
  pool_n <- pool_multiplier * n_cells
  set.seed(seed)
  splatter::splatSimulateGroups(
    batchCells = c(pool_n, pool_n),
    nGenes = n_genes,
    group.prob = GROUP_PROBABILITY,
    # 10% of genes, 2x, upward only: the hand-built perturbation, expressed in
    # splatter's parameters.
    de.prob = c(0.0, 0.10),
    de.downProb = c(0.0, 0.0),
    de.facLoc = c(0.0, log(2.0)),
    de.facScale = c(0.0, 0.10),
    # No batch effect: the two batches exist only to give the two sides
    # independent draws from one population.
    batch.facLoc = 0.0,
    batch.facScale = 0.0,
    lib.loc = log(median_depth),
    lib.scale = sigma,
    seed = seed,
    verbose = FALSE
  )
}

take <- function(available, wanted, what) {
  if (length(available) < wanted) {
    stop(sprintf("pool holds %d %s cells, need %d; raise the pool multiplier",
                 length(available), what, wanted))
  }
  available[seq_len(wanted)]
}

write_side <- function(counts, path) {
  # Cells x genes, matching the Python runner's orientation. Splatter is
  # genes x cells, so this transposes.
  Matrix::writeMM(as(Matrix::Matrix(t(counts), sparse = TRUE), "CsparseMatrix"),
                  file = path)
}

dir.create(output_root, recursive = TRUE, showWarnings = FALSE)
records <- list()
failures <- character(0)

for (replicate in seq_len(replicates)) {
  for (sigma in sigmas) {
    for (kind in c("homogeneous", "perturbed")) {
      # The perturbed arm is the positive control and lives at uniform depth,
      # exactly as in the hand-built screen, so it is generated once.
      if (kind == "perturbed" && !isTRUE(all.equal(sigma, 0.0))) next
      arm <- if (kind == "homogeneous") homogeneous_name(sigma) else "perturbed_depth_cv0"

      seed <- base_seed + 7919L * replicate + as.integer(round(1000 * sigma)) +
        if (kind == "perturbed") 101L else 0L
      simulation <- simulate_pool(sigma, seed)
      pool <- SingleCellExperiment::counts(simulation)
      metadata <- as.data.frame(SummarizedExperiment::colData(simulation))
      batch <- as.character(metadata$Batch)
      group <- as.integer(sub("Group", "", as.character(metadata$Group), fixed = TRUE))
      batches <- unique(batch)

      first_shared <- which(batch == batches[[1L]] & group == 1L)
      first_extra <- which(batch == batches[[1L]] & group == 2L)
      second_shared <- which(batch == batches[[2L]] & group == 1L)

      if (kind == "homogeneous") {
        perturbed_n <- 0L
      } else {
        perturbed_n <- as.integer(round(perturbed_fraction * n_cells))
      }
      shared_n <- n_cells - perturbed_n
      source_index <- c(take(first_shared, shared_n, "shared"),
                        if (perturbed_n > 0L) take(first_extra, perturbed_n, "perturbed"))
      target_index <- take(second_shared, n_cells, "target shared")
      perturbed_mask <- c(rep(0L, shared_n), rep(1L, perturbed_n))

      arm_directory <- file.path(output_root, arm, sprintf("rep_%02d", replicate))
      dir.create(arm_directory, recursive = TRUE, showWarnings = FALSE)
      write_side(pool[, source_index, drop = FALSE],
                 file.path(arm_directory, "source.mtx"))
      write_side(pool[, target_index, drop = FALSE],
                 file.path(arm_directory, "target.mtx"))
      utils::write.csv(
        data.frame(perturbed = perturbed_mask),
        file.path(arm_directory, "source_perturbed.csv"), row.names = FALSE
      )

      # Did splatter's lib.scale do what its documentation says? Measured on
      # the realised counts, both sides pooled.
      realised <- c(Matrix::colSums(pool[, source_index, drop = FALSE]),
                    Matrix::colSums(pool[, target_index, drop = FALSE]))
      realised <- realised[realised > 0]
      observed_sigma <- stats::sd(log(realised))
      observed_cv <- stats::sd(realised) / mean(realised)
      expected_cv <- sqrt(exp(sigma^2) - 1)
      deviation <- abs(observed_sigma - sigma)
      status <- if (deviation <= tolerance) "ok" else "MISMATCH"
      if (status != "ok") {
        failures <- c(failures, sprintf(
          "%s rep %d: asked lib.scale=%.3f, realised sd(log depth)=%.3f",
          arm, replicate, sigma, observed_sigma))
      }
      cat(sprintf(
        "  %-28s rep %d  lib.scale=%.2f  sd(log depth)=%.3f  CV=%.3f (expected %.3f)  median=%.0f  %s\n",
        arm, replicate, sigma, observed_sigma, observed_cv, expected_cv,
        stats::median(realised), status))

      records[[length(records) + 1L]] <- data.frame(
        arm = arm, replicate = replicate, lib_scale = sigma,
        requested_median_depth = median_depth,
        observed_sd_log_depth = observed_sigma,
        observed_cv = observed_cv, expected_cv = expected_cv,
        observed_median_depth = stats::median(realised),
        source_cells = length(source_index), target_cells = length(target_index),
        perturbed_cells = perturbed_n, genes = nrow(pool),
        seed = seed, status = status, stringsAsFactors = FALSE
      )
    }
  }
}

manifest <- do.call(rbind, records)
utils::write.csv(manifest, file.path(output_root, "splatter_depth_manifest.csv"),
                 row.names = FALSE)
cat(sprintf("\nwrote %s\n", file.path(output_root, "splatter_depth_manifest.csv")))
cat(sprintf("splatter %s\n", as.character(utils::packageVersion("splatter"))))

if (length(failures) > 0L) {
  cat("\n", length(failures), " arm(s) did not realise the requested spread:\n", sep = "")
  for (line in failures) cat("  ", line, "\n", sep = "")
  stop("splatter's lib.scale did not produce the requested depth ladder")
}
cat("every arm realised the requested depth spread\n")
