#!/usr/bin/env Rscript
# Ask Splatter two questions before building a benchmark on the answers.
#
# The existing generator records "Splatter has one global library size and no
# per-batch override" and thins the target's reads downstream because of it.
# That claim decides the whole design, so it is asked of the installed package
# rather than assumed, and the version that answered is printed.
#
# **Question 1. Does this Splatter accept a per-batch library size?**
# If `lib.loc` takes a vector of length nBatches, one simulation can carry both
# sides at different depths and nothing downstream has to touch the counts.
# setParams is the authority: it runs the class validity check.
#
# **Question 2. If not, do two calls with the same seed share their biology?**
# Two single-batch calls differing only in `lib.loc` would give "the same
# biology observed at two depths" -- but only if the gene means and the group
# assignment come out identical. They should: `lib.loc` is a parameter value,
# not a count, so it cannot change how many random numbers are drawn before
# the gene means, and an equal number of draws leaves the stream at the same
# place. That is a prediction about Splatter's internals, so it is measured.
# If the gene means differ, two calls introduce biological difference between
# the sides and the design is unusable -- which is the failure this probe
# exists to catch before a night of generation, not after.
#
# Small on purpose: 200 cells, 500 genes. Both questions are about parameter
# handling and RNG order, and neither needs scale.
#
# Usage:
#   Rscript benchmark/tacc/09_probe_splatter_library.R [--out DIR]

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
out <- value_of("--out", NULL)

N <- 200L
G <- 500L
SEED <- 20260930L
LIB_LOC <- 11.0
DEPTH_RATIO <- 0.50
LIB_LOC_LOW <- LIB_LOC + log(DEPTH_RATIO)

cat("================ versions ================\n")
cat(sprintf("splatter   %s\n", as.character(packageVersion("splatter"))))
cat(sprintf("R          %s\n", R.version.string))
cat("\n")

# ---- question 1: is lib.loc per batch ------------------------------------
cat("======== Q1: per-batch lib.loc accepted ========\n")
base <- setParams(newSplatParams(), nGenes = G, batchCells = c(N, N),
                  batch.facLoc = 0, batch.facScale = 0, seed = SEED)
vector_ok <- tryCatch({
  probe <- setParams(base, lib.loc = c(LIB_LOC, LIB_LOC_LOW))
  got <- getParam(probe, "lib.loc")
  cat(sprintf("  setParams accepted it; lib.loc is now length %d: %s\n",
              length(got), paste(sprintf("%.4f", got), collapse = ", ")))
  # Accepted is not the same as honoured: a validity check that silently keeps
  # the first element would pass here and then simulate one depth.
  length(got) == 2L
}, error = function(e) {
  cat(sprintf("  REFUSED: %s\n", conditionMessage(e)))
  FALSE
})
cat(sprintf("  lib.loc is per batch: %s\n", vector_ok))
if (vector_ok) {
  sim <- splatSimulate(setParams(base, lib.loc = c(LIB_LOC, LIB_LOC_LOW)),
                       method = "groups", verbose = FALSE)
  depth <- tapply(Matrix::colSums(counts(sim)), colData(sim)$Batch, median)
  cat("  realised median depth per batch:\n")
  for (nm in names(depth)) cat(sprintf("    %-8s %10.0f\n", nm, depth[[nm]]))
  cat(sprintf("  ratio Batch2/Batch1 = %.4f (asked %.4f)\n",
              depth[[2]] / depth[[1]], DEPTH_RATIO))
}
cat("\n")

# ---- question 2: do two same-seed calls share their biology ---------------
cat("==== Q2: two calls, same seed, different lib.loc ====\n")
one_side <- function(lib_loc, dropout_mid = NULL) {
  params <- setParams(newSplatParams(), nGenes = G, batchCells = N,
                      batch.facLoc = 0, batch.facScale = 0,
                      group.prob = rep(0.2, 5), lib.loc = lib_loc,
                      seed = SEED)
  if (is.null(dropout_mid)) {
    params <- setParams(params, dropout.type = "none")
  } else {
    params <- setParams(params, dropout.mid = dropout_mid, dropout.shape = -1)
    params <- setParams(params, dropout.type = "experiment")
  }
  splatSimulate(params, method = "groups", verbose = FALSE)
}

high <- one_side(LIB_LOC)
low <- one_side(LIB_LOC_LOW)

means_high <- as.numeric(rowData(high)$GeneMean)
means_low <- as.numeric(rowData(low)$GeneMean)
groups_high <- as.character(colData(high)$Group)
groups_low <- as.character(colData(low)$Group)

means_identical <- isTRUE(all.equal(means_high, means_low, tolerance = 0))
groups_identical <- identical(groups_high, groups_low)
cat(sprintf("  gene means identical:      %s  (max abs diff %.3e)\n",
            means_identical, max(abs(means_high - means_low))))
cat(sprintf("  group assignment identical: %s  (%d of %d cells agree)\n",
            groups_identical, sum(groups_high == groups_low), N))

d_high <- Matrix::colSums(counts(high))
d_low <- Matrix::colSums(counts(low))
f_high <- Matrix::colSums(counts(high) > 0)
f_low <- Matrix::colSums(counts(low) > 0)
cat(sprintf("  median depth   high %8.0f   low %8.0f   ratio %.4f (asked %.4f)\n",
            median(d_high), median(d_low), median(d_low) / median(d_high),
            DEPTH_RATIO))
cat(sprintf("  median nFeature high %8.0f   low %8.0f   ratio %.4f\n",
            median(f_high), median(f_low), median(f_low) / median(f_high)))
cat("\n")

# ---- and with dropout on the low side, which is what L2 needs ------------
cat("==== Q2b: low depth plus Splatter dropout ====\n")
low_drop <- one_side(LIB_LOC_LOW, dropout_mid = 1.0)
means_drop <- as.numeric(rowData(low_drop)$GeneMean)
cat(sprintf("  gene means still identical to the high side: %s\n",
            isTRUE(all.equal(means_high, means_drop, tolerance = 0))))
d_drop <- Matrix::colSums(counts(low_drop))
f_drop <- Matrix::colSums(counts(low_drop) > 0)
cat(sprintf("  median depth    %8.0f  (ratio to high %.4f)\n",
            median(d_drop), median(d_drop) / median(d_high)))
cat(sprintf("  median nFeature %8.0f  (ratio to high %.4f)\n",
            median(f_drop), median(f_drop) / median(f_high)))
cat(sprintf("  nFeature falls further than depth: %s\n",
            (median(f_drop) / median(f_high)) < (median(d_drop) / median(d_high))))
cat("\n")

cat("================ verdict ================\n")
if (vector_ok) {
  cat("ROUTE A: one simulation, two batches, per-batch lib.loc.\n")
} else if (means_identical && groups_identical) {
  cat("ROUTE B: two single-batch calls sharing seed and biology,\n")
  cat("         differing only in lib.loc. The biology is identical, so the\n")
  cat("         depth difference is the only difference between the sides.\n")
} else {
  cat("NEITHER ROUTE IS SAFE. Two calls do not share their biology, so the\n")
  cat("sides would differ biologically and the benchmark's ground truth\n")
  cat("('every population is on both sides') would not hold. Stop here.\n")
}

if (!is.null(out)) {
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  record <- list(
    splatter_version = as.character(packageVersion("splatter")),
    r_version = R.version.string,
    per_batch_lib_loc = vector_ok,
    two_call_gene_means_identical = means_identical,
    two_call_groups_identical = groups_identical,
    asked_depth_ratio = DEPTH_RATIO,
    realised_depth_ratio = as.numeric(median(d_low) / median(d_high)),
    realised_nfeature_ratio = as.numeric(median(f_low) / median(f_high)),
    dropout_depth_ratio = as.numeric(median(d_drop) / median(d_high)),
    dropout_nfeature_ratio = as.numeric(median(f_drop) / median(f_high)),
    route = if (vector_ok) "A" else if (means_identical && groups_identical) "B" else "NONE"
  )
  writeLines(jsonlite::toJSON(record, auto_unbox = TRUE, pretty = TRUE),
             file.path(out, "splatter_library_probe.json"))
  cat(sprintf("\nwrote %s\n", file.path(out, "splatter_library_probe.json")))
}
