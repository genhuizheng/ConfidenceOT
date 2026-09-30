#!/usr/bin/env Rscript
# Splatter generation where the depth mismatch is Splatter's own.
#
# The earlier generator could not express the depth difference: Splatter's
# `lib.loc` is an experiment-level parameter, so one simulation carrying both
# sides as two batches gives both sides one library-size distribution. The
# mismatch was therefore made downstream, by binomial thinning of the target's
# counts. That is the exact likelihood of sequencing the same library less
# deeply, and it is still a post-hoc operation on counts -- and on this
# project's own reading it is worse than that, because the read-equalisation
# arm inverts precisely that mechanism, so an equalised L1 result was never
# independent evidence.
#
# Here the depth comes from the simulator. Which route that takes is asked of
# the installed Splatter rather than assumed, and the answer is recorded:
#
# **Route A** -- if `lib.loc` accepts one value per batch, one simulation
# carries both sides at two depths and nothing else is needed.
#
# **Route B** -- otherwise, two single-batch calls with the same seed and the
# same biological parameters, differing only in `lib.loc`. Checked and not
# trusted: the gene means, the **per-group DE factors** and the group labels
# must come out identical or this script stops. Marginal gene means agreeing
# says nothing about the group-specific parameters, and those are what make
# the populations distinct.
#
# Route B also needs `--allow-shared-seed-sides`, and the flag exists because
# of what one seed does beyond sharing the biology: cell i in both calls draws
# the same group, the same DE factors and the same BCV value from the same
# stream position, so the two differ only by a library factor. That is a
# near-duplicate pair at two depths, and it hands the transport a trivially
# correct answer that has nothing to do with the biology under test.
# 09_probe_splatter_library.R measures whether that pairing appeared; the flag
# says an operator read the answer. It is not a default.
#
# The depth ratio is set through the log-normal's median, which is exactly
# `exp(lib.loc)`:
#
#     lib.loc_target = lib.loc_source + log(depth_ratio)
#
# so asking for 0.50 asks for half the median library, in the simulator's own
# parameterisation. The realised ratio is measured and written out, because a
# log-normal's realised median is not its parameter.
#
# Dropout stays Splatter's own, on the target side only, and only at L2.
#
# Writes Matrix Market rather than h5ad, as the previous generator did and for
# the same reason: no zellkonverter or anndata in R on this account.
#
# Usage:
#   Rscript 10_generate_splatter_native.R --out DIR --n 1000 --replicate 1 \
#       --level L0_matched --seed 20260930 \
#       [--groups 5] [--lib-loc 11.0] [--depth-ratio 0.50] \
#       [--dropout-mid-target 1.0] [--dropout-shape -1]

suppressPackageStartupMessages({
  library(splatter)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
value_of <- function(flag, default = NULL) {
  hit <- which(args == flag)
  if (length(hit) == 0) return(default)
  if (hit[1] == length(args)) stop(sprintf("%s needs a value", flag))
  args[hit[1] + 1]
}

out <- value_of("--out")
if (is.null(out)) stop("--out is required")
n_cells <- as.integer(value_of("--n", "1000"))
replicate <- as.integer(value_of("--replicate", "1"))
level <- value_of("--level", "L0_matched")
groups <- as.integer(value_of("--groups", "5"))
seed <- as.integer(value_of("--seed", "20260930"))
n_genes <- as.integer(value_of("--n-genes", "0"))
lib_loc <- as.numeric(value_of("--lib-loc", "11.0"))
lib_scale <- as.numeric(value_of("--lib-scale", "0.2"))
depth_ratio <- as.numeric(value_of("--depth-ratio", "0.50"))
dropout_mid_target <- as.numeric(value_of("--dropout-mid-target", "1.0"))
dropout_shape <- as.numeric(value_of("--dropout-shape", "-1"))
allow_shared_seed <- "--allow-shared-seed-sides" %in% args

LEVELS <- c("L0_matched", "L1_depth", "L2_depth_dropout")
if (!(level %in% LEVELS)) {
  stop(sprintf("--level must be one of %s", paste(LEVELS, collapse = ", ")))
}

# What each level asks of the target side, and nothing else differs.
#   L0  matched depth, no dropout
#   L1  lower depth, no dropout
#   L2  lower depth and Splatter dropout
target_ratio <- if (level == "L0_matched") 1.0 else depth_ratio
target_dropout <- if (level == "L2_depth_dropout") dropout_mid_target else NA_real_
lib_loc_target <- lib_loc + log(target_ratio)

dir.create(out, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("level %s: source lib.loc %.4f, target lib.loc %.4f (ratio %.4f)\n",
            level, lib_loc, lib_loc_target, target_ratio))
cat(sprintf("  target dropout.mid %s\n",
            if (is.na(target_dropout)) "none" else sprintf("%.4f", target_dropout)))

base_params <- function(cells_per_batch) {
  params <- setParams(
    newSplatParams(),
    batchCells     = cells_per_batch,
    batch.facLoc   = 0,   # no batch effect: the sides must differ only by
    batch.facScale = 0,   #   the observation setting
    group.prob     = rep(1 / groups, groups),
    lib.scale      = lib_scale,
    seed           = seed
  )
  if (n_genes > 0) params <- setParams(params, nGenes = n_genes)
  params
}

# ---- which route does this Splatter allow --------------------------------
per_batch_lib_loc <- tryCatch({
  probe <- setParams(base_params(c(n_cells, n_cells)),
                     lib.loc = c(lib_loc, lib_loc_target))
  length(getParam(probe, "lib.loc")) == 2L
}, error = function(e) {
  cat(sprintf("  per-batch lib.loc refused: %s\n", conditionMessage(e)))
  FALSE
})

assertions <- list(gene_means_identical = NA, groups_identical = NA,
                   group_de_identical = NA, group_de_worst_abs_diff = NA,
                   group_de_columns = NA,
                   shared_seed_sides_allowed = allow_shared_seed)

if (per_batch_lib_loc) {
  route <- "A"
  cat("  route A: one simulation, per-batch lib.loc\n")
  params <- setParams(base_params(c(n_cells, n_cells)),
                      lib.loc = c(lib_loc, lib_loc_target))
  if (is.na(target_dropout)) {
    params <- setParams(params, dropout.type = "none")
  } else {
    # A very low midpoint with a negative shape puts the source logistic
    # essentially at zero, so the source side is left untouched.
    params <- setParams(params,
                        dropout.mid = c(-10, target_dropout),
                        dropout.shape = c(dropout_shape, dropout_shape))
    params <- setParams(params, dropout.type = "batch")
  }
  sim <- splatSimulate(params, method = "groups", verbose = FALSE)
  counts_all <- counts(sim)
  batch_labels <- as.character(colData(sim)$Batch)
  # Splatter names them Batch1 and Batch2; the Python step takes the first
  # by sort order as the source, so the names have to sort that way.
  batch_labels <- ifelse(batch_labels == sort(unique(batch_labels))[1],
                         "Batch1", "Batch2")
  cell_ids <- paste0(batch_labels, "_", colnames(sim))
  group_labels <- as.character(colData(sim)$Group)
  gene_ids <- rownames(sim)
  gene_means <- as.numeric(rowData(sim)$GeneMean)
} else {
  route <- "B"
  cat("  route B: two single-batch calls, same seed, different lib.loc\n")
  # The gate, before any generation. One seed pairs the sides cell by cell as
  # well as sharing their biology, and a paired benchmark measures nothing.
  if (!allow_shared_seed) {
    stop(paste0(
      "this Splatter has no per-batch lib.loc, so the only native route is ",
      "two calls sharing one seed -- and that seed pairs the sides cell by ",
      "cell as well as sharing their biology, which would hand the transport ",
      "a trivially correct answer. Run ",
      "benchmark/tacc/09_probe_splatter_library.R first; if it reports ",
      "route B (no invented pairing), pass --allow-shared-seed-sides. If it ",
      "reports B_PAIRED, do not generate on this route."))
  }
  one_side <- function(this_lib_loc, this_dropout) {
    params <- setParams(base_params(n_cells), lib.loc = this_lib_loc)
    if (is.na(this_dropout)) {
      params <- setParams(params, dropout.type = "none")
    } else {
      params <- setParams(params, dropout.mid = this_dropout,
                          dropout.shape = dropout_shape)
      params <- setParams(params, dropout.type = "experiment")
    }
    splatSimulate(params, method = "groups", verbose = FALSE)
  }
  source_sim <- one_side(lib_loc, NA_real_)
  target_sim <- one_side(lib_loc_target, target_dropout)

  # The check the whole route rests on. Not a warning: a difference here means
  # the two sides are biologically different simulations, and every later
  # statement about "the same biology at two depths" would be false.
  source_means <- as.numeric(rowData(source_sim)$GeneMean)
  target_means <- as.numeric(rowData(target_sim)$GeneMean)
  source_groups <- as.character(colData(source_sim)$Group)
  target_groups <- as.character(colData(target_sim)$Group)
  assertions$gene_means_identical <- identical(source_means, target_means)
  assertions$groups_identical <- identical(source_groups, target_groups)

  # The group-specific parameters, separately. Agreeing marginal gene means
  # say nothing about the per-group DE factors, and those are what make the
  # populations distinct -- so "every population is on both sides" rests on
  # these agreeing, not on the means.
  de_cols <- grep("^DEFacGroup", colnames(rowData(source_sim)), value = TRUE)
  de_identical <- length(de_cols) > 0
  de_worst <- 0
  for (nm in de_cols) {
    a <- as.numeric(rowData(source_sim)[[nm]])
    b <- as.numeric(rowData(target_sim)[[nm]])
    de_identical <- de_identical && identical(a, b)
    de_worst <- max(de_worst, max(abs(a - b)))
  }
  assertions$group_de_identical <- de_identical
  assertions$group_de_worst_abs_diff <- de_worst
  assertions$group_de_columns <- length(de_cols)
  if (!de_identical) {
    stop(sprintf(paste0(
      "two calls with seed %d produced different group DE factors across %d ",
      "column(s) (worst abs diff %.3e). The populations would not be the same ",
      "populations on the two sides. Refusing to generate."),
      seed, length(de_cols), de_worst))
  }
  if (!identical(rownames(source_sim), rownames(target_sim))) {
    stop("the two calls disagree on gene names, so their columns cannot be ",
         "placed in one matrix")
  }
  if (!assertions$gene_means_identical) {
    stop(sprintf(paste0(
      "two calls with seed %d produced different gene means (max abs diff ",
      "%.3e). The sides would differ biologically, so the benchmark's ground ",
      "truth that every population is on both sides would not hold. Refusing ",
      "to generate."), seed, max(abs(source_means - target_means))))
  }
  if (!assertions$groups_identical) {
    stop(sprintf(paste0(
      "two calls with seed %d assigned different groups (%d of %d cells ",
      "agree). Refusing to generate."), seed,
      sum(source_groups == target_groups), n_cells))
  }
  cat("  shared biology confirmed: gene means and group labels identical\n")

  counts_all <- cbind(counts(source_sim), counts(target_sim))
  batch_labels <- c(rep("Batch1", ncol(source_sim)),
                    rep("Batch2", ncol(target_sim)))
  cell_ids <- c(paste0("Batch1_", colnames(source_sim)),
                paste0("Batch2_", colnames(target_sim)))
  group_labels <- c(source_groups, target_groups)
  gene_ids <- rownames(source_sim)
  gene_means <- source_means
}

colnames(counts_all) <- cell_ids

# ---- what actually came out ----------------------------------------------
depth <- Matrix::colSums(counts_all)
detected <- Matrix::colSums(counts_all > 0)
is_source <- batch_labels == "Batch1"
realised <- list(
  source_median_nCount = as.numeric(median(depth[is_source])),
  target_median_nCount = as.numeric(median(depth[!is_source])),
  source_median_nFeature = as.numeric(median(detected[is_source])),
  target_median_nFeature = as.numeric(median(detected[!is_source])),
  source_min_nFeature = as.integer(min(detected[is_source])),
  target_min_nFeature = as.integer(min(detected[!is_source]))
)
realised$realised_depth_ratio <-
  realised$target_median_nCount / realised$source_median_nCount
realised$realised_nfeature_ratio <-
  realised$target_median_nFeature / realised$source_median_nFeature
cat(sprintf("  realised depth   source %8.0f  target %8.0f  ratio %.4f\n",
            realised$source_median_nCount, realised$target_median_nCount,
            realised$realised_depth_ratio))
cat(sprintf("  realised nFeature source %8.0f  target %8.0f  ratio %.4f\n",
            realised$source_median_nFeature, realised$target_median_nFeature,
            realised$realised_nfeature_ratio))

cells <- data.frame(cell_id = cell_ids, batch = batch_labels,
                    group = group_labels, stringsAsFactors = FALSE)
genes <- data.frame(gene_id = gene_ids, gene_mean = gene_means,
                    stringsAsFactors = FALSE)

writeMM(as(counts_all, "CsparseMatrix"), file.path(out, "counts.mtx"))
write.csv(cells, file.path(out, "cells.csv"), row.names = FALSE)
write.csv(genes, file.path(out, "genes.csv"), row.names = FALSE)

record <- c(
  realised,
  assertions,
  list(
    parameter_source = "splatter defaults with lib.loc set per side",
    depth_mechanism = "splatter_library_size",
    route = route,
    per_batch_lib_loc = per_batch_lib_loc,
    n_cells_per_side = n_cells,
    groups = groups,
    replicate = replicate,
    technical_level = level,
    seed = seed,
    lib_loc_source = lib_loc,
    lib_loc_target = lib_loc_target,
    lib_scale = lib_scale,
    asked_depth_ratio = target_ratio,
    dropout_type = if (is.na(target_dropout)) "none" else "target only",
    dropout_mid_target = target_dropout,
    dropout_shape = if (is.na(target_dropout)) NA_real_ else dropout_shape,
    batch_fac_loc = 0,
    batch_fac_scale = 0,
    independent_realization = TRUE,
    splatter_version = as.character(packageVersion("splatter")),
    r_version = R.version.string
  )
)
writeLines(jsonlite::toJSON(record, auto_unbox = TRUE, pretty = TRUE,
                            null = "null", na = "null"),
           file.path(out, "generation.json"))
writeLines("SUCCESS", file.path(out, "GENERATION_DONE"))
cat(sprintf("wrote %s (%d cells per side, %d groups, level %s, route %s)\n",
            out, n_cells, groups, level, route))
