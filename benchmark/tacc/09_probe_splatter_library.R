#!/usr/bin/env Rscript
# Ask Splatter whether a native per-side library size is available, and if not,
# whether two calls can stand in for one. Three questions, no more.
#
# The existing generator records "Splatter has one global library size and no
# per-batch override" and thins the target's reads downstream because of it.
# That claim decides the whole design, so it is asked of the installed package
# rather than assumed, and the version that answered is printed.
#
# **Q1. Does this Splatter accept a per-batch library size?**
# If `lib.loc` takes one value per batch, one simulation carries both sides at
# two depths, splatter draws each batch's cells independently, and nothing
# downstream touches the counts. Accepted is not the same as honoured, so the
# realised per-batch depths are measured too.
#
# **Q2. If not, can two calls share the biology without inventing a pairing?**
# Two single-batch calls with one seed differing only in `lib.loc` would give
# "the same biology at two depths" -- but only if three things hold, and the
# third is the one that can quietly ruin the benchmark:
#
#   a. the gene means agree;
#   b. the **group-specific** parameters agree -- the per-group DE factors are
#      what make the populations distinct, and marginal gene means agreeing
#      says nothing about them;
#   c. there is **no per-cell correspondence between the sides**. One seed
#      means cell i in both calls draws the same group, the same DE factors
#      and the same BCV value from the same stream position, so the two
#      differ only by a library factor. That is a near-duplicate pair at two
#      depths, and it hands the transport a trivially correct answer that has
#      nothing to do with the biology the benchmark claims to test. It is
#      measured directly: how often a source cell's nearest target is its own
#      index, and where the same-index target ranks among all targets.
#
# **Q3. Do a native library size and native dropout compose for L2?**
# Reported as what was set and what changed in detection, not as a threshold
# to pass: whether nFeature falls further than depth is a prediction about
# Splatter's dropout, and a probe that gates on its own prediction cannot
# report a surprise.
#
# Small on purpose: 200 cells, 500 genes. Every question is about parameter
# handling and RNG order, and none of them needs scale.
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
K <- 5L
SEED <- 20260930L
LIB_LOC <- 11.0
DEPTH_RATIO <- 0.50
LIB_LOC_LOW <- LIB_LOC + log(DEPTH_RATIO)

cat("================ versions ================\n")
cat(sprintf("splatter   %s\n", as.character(packageVersion("splatter"))))
cat(sprintf("R          %s\n", R.version.string))
cat("\n")

# ---- Q1: is lib.loc per batch -------------------------------------------
cat("======== Q1: per-batch lib.loc ========\n")
base_two <- setParams(newSplatParams(), nGenes = G, batchCells = c(N, N),
                      batch.facLoc = 0, batch.facScale = 0,
                      group.prob = rep(1 / K, K), lib.scale = 0.2,
                      seed = SEED)
vector_kept <- tryCatch({
  probe <- setParams(base_two, lib.loc = c(LIB_LOC, LIB_LOC_LOW))
  got <- getParam(probe, "lib.loc")
  cat(sprintf("  setParams accepted it; lib.loc is length %d: %s\n",
              length(got), paste(sprintf("%.4f", got), collapse = ", ")))
  length(got) == 2L
}, error = function(e) {
  cat(sprintf("  REFUSED: %s\n", conditionMessage(e)))
  FALSE
})

route_a_honoured <- FALSE
route_a <- list()
if (vector_kept) {
  sim <- splatSimulate(setParams(base_two, lib.loc = c(LIB_LOC, LIB_LOC_LOW)),
                       method = "groups", verbose = FALSE)
  d <- Matrix::colSums(counts(sim))
  b <- as.character(colData(sim)$Batch)
  hi <- median(d[b == sort(unique(b))[1]])
  lo <- median(d[b == sort(unique(b))[2]])
  route_a$batch1_median_nCount <- as.numeric(hi)
  route_a$batch2_median_nCount <- as.numeric(lo)
  route_a$realised_depth_ratio <- as.numeric(lo / hi)
  cat(sprintf("  realised medians  batch1 %8.0f  batch2 %8.0f  ratio %.4f (asked %.4f)\n",
              hi, lo, lo / hi, DEPTH_RATIO))
  # Honoured means the two batches really came out at two depths. A validity
  # check that kept the vector but simulated from its first element would
  # give a ratio of one here.
  route_a_honoured <- abs(lo / hi - DEPTH_RATIO) < 0.15
  cat(sprintf("  honoured (ratio within 0.15 of asked): %s\n", route_a_honoured))
}
cat(sprintf("  Q1 answer: per-batch lib.loc usable = %s\n", route_a_honoured))
cat("\n")

# ---- Q2: two calls, one seed --------------------------------------------
cat("==== Q2: two calls, same seed, different lib.loc ====\n")
one_side <- function(lib_loc, dropout_mid = NULL, seed = SEED) {
  params <- setParams(newSplatParams(), nGenes = G, batchCells = N,
                      batch.facLoc = 0, batch.facScale = 0,
                      group.prob = rep(1 / K, K), lib.loc = lib_loc,
                      lib.scale = 0.2, seed = seed)
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

# (a) marginal gene means
means_high <- as.numeric(rowData(high)$GeneMean)
means_low <- as.numeric(rowData(low)$GeneMean)
means_identical <- identical(means_high, means_low)
cat(sprintf("  (a) gene means identical:        %s   (max abs diff %.3e)\n",
            means_identical, max(abs(means_high - means_low))))

# (b) group-specific parameters. The per-group DE factors are what make the
# populations distinct; agreeing marginal means say nothing about them.
de_cols <- grep("^DEFacGroup", colnames(rowData(high)), value = TRUE)
cat(sprintf("      group DE factor columns found: %s\n",
            if (length(de_cols)) paste(de_cols, collapse = ", ") else "NONE"))
de_identical <- length(de_cols) > 0
de_worst <- 0
for (nm in de_cols) {
  a <- as.numeric(rowData(high)[[nm]])
  b <- as.numeric(rowData(low)[[nm]])
  same <- identical(a, b)
  de_worst <- max(de_worst, max(abs(a - b)))
  de_identical <- de_identical && same
  cat(sprintf("      %-14s identical %s  (max abs diff %.3e)\n",
              nm, same, max(abs(a - b))))
}
groups_high <- as.character(colData(high)$Group)
groups_low <- as.character(colData(low)$Group)
groups_identical <- identical(groups_high, groups_low)
cat(sprintf("  (b) group DE factors identical:  %s\n", de_identical))
cat(sprintf("      per-cell group labels identical: %s (%d of %d agree)\n",
            groups_identical, sum(groups_high == groups_low), N))

# (c) the pairing the shared seed may have invented. Compared in log-CPM so
# the library-size difference the two sides are supposed to have is divided
# out and what is left is whether cell i resembles cell i more than chance.
lcpm <- function(sim) {
  m <- as.matrix(counts(sim))
  total <- pmax(colSums(m), 1)
  log1p(t(t(m) / total) * 1e4)
}
A <- lcpm(high)
B <- lcpm(low)
# squared euclidean between every source cell and every target cell
sq <- outer(colSums(A^2), colSums(B^2), "+") - 2 * crossprod(A, B)
nearest <- apply(sq, 1, which.min)
self_nearest <- mean(nearest == seq_len(N))
# where the same-index target sits among all targets, 0 = closest
self_rank <- vapply(seq_len(N), function(i) {
  (sum(sq[i, ] < sq[i, i]) ) / (N - 1)
}, numeric(1))
cat(sprintf("  (c) source cell's nearest target is its own index: %.1f%% of cells\n",
            100 * self_nearest))
cat(sprintf("      same-index target's mean normalised rank: %.4f (chance 0.5)\n",
            mean(self_rank)))
cat(sprintf("      median normalised rank: %.4f\n", median(self_rank)))
# Chance is 1/N for the nearest and 0.5 for the mean rank. Anything far below
# 0.5 is an invented pairing, whatever the gene means say.
pairing_free <- self_nearest < (5 / N) && mean(self_rank) > 0.40
cat(sprintf("      no invented pairing: %s\n", pairing_free))

d_high <- Matrix::colSums(counts(high))
d_low <- Matrix::colSums(counts(low))
f_high <- Matrix::colSums(counts(high) > 0)
f_low <- Matrix::colSums(counts(low) > 0)
cat(sprintf("      realised depth   high %8.0f  low %8.0f  ratio %.4f (asked %.4f)\n",
            median(d_high), median(d_low), median(d_low) / median(d_high),
            DEPTH_RATIO))
cat("\n")

# ---- Q2d: does a different seed break the pairing, and at what cost ------
cat("==== Q2d: different seed on the target side ====\n")
low_other <- one_side(LIB_LOC_LOW, seed = SEED + 1L)
means_other <- as.numeric(rowData(low_other)$GeneMean)
other_means_identical <- identical(means_high, means_other)
Bo <- lcpm(low_other)
sqo <- outer(colSums(A^2), colSums(Bo^2), "+") - 2 * crossprod(A, Bo)
nearest_o <- apply(sqo, 1, which.min)
self_nearest_o <- mean(nearest_o == seq_len(N))
cat(sprintf("  gene means identical to the high side: %s (max abs diff %.3e)\n",
            other_means_identical, max(abs(means_high - means_other))))
cat(sprintf("  nearest target is own index: %.1f%% of cells\n",
            100 * self_nearest_o))
cat("  A different seed removes the pairing and the shared biology together,\n")
cat("  so it is reported, not recommended.\n")
cat("\n")

# ---- Q3: native library size and native dropout together ----------------
cat("==== Q3: L2, native lib.loc plus native dropout ====\n")
low_drop <- one_side(LIB_LOC_LOW, dropout_mid = 1.0)
drop_params <- setParams(newSplatParams(), nGenes = G, batchCells = N,
                         lib.loc = LIB_LOC_LOW, lib.scale = 0.2, seed = SEED)
drop_params <- setParams(drop_params, dropout.mid = 1.0, dropout.shape = -1)
drop_params <- setParams(drop_params, dropout.type = "experiment")
cat(sprintf("  dropout.type  %s\n", getParam(drop_params, "dropout.type")))
cat(sprintf("  dropout.mid   %s\n",
            paste(getParam(drop_params, "dropout.mid"), collapse = ", ")))
cat(sprintf("  dropout.shape %s\n",
            paste(getParam(drop_params, "dropout.shape"), collapse = ", ")))
cat(sprintf("  Dropout column present in assays: %s\n",
            "Dropout" %in% assayNames(low_drop)))
d_drop <- Matrix::colSums(counts(low_drop))
f_drop <- Matrix::colSums(counts(low_drop) > 0)
cat(sprintf("  detection rate  L0-equivalent %.4f   L1-equivalent %.4f   L2 %.4f\n",
            median(f_high) / G, median(f_low) / G, median(f_drop) / G))
cat(sprintf("  median nFeature high %6.0f  low %6.0f  low+dropout %6.0f\n",
            median(f_high), median(f_low), median(f_drop)))
cat(sprintf("  median depth    high %8.0f  low %8.0f  low+dropout %8.0f\n",
            median(d_high), median(d_low), median(d_drop)))
cat(sprintf("  dropout changed detection relative to the same depth: %.4f -> %.4f\n",
            median(f_low) / G, median(f_drop) / G))
# The observation, not a gate. Whether nFeature falls further than depth is a
# prediction about Splatter's dropout model; it is printed so a surprise is
# visible rather than converted into a pass or a fail.
cat(sprintf("  for reference, depth ratio %.4f vs nFeature ratio %.4f\n",
            median(d_drop) / median(d_high), median(f_drop) / median(f_high)))
l2_composes <- median(f_drop) < median(f_low)
cat(sprintf("  Q3 answer: dropout composes with the native depth = %s\n",
            l2_composes))
cat("\n")

cat("================ verdict ================\n")
route <- "NONE"
if (route_a_honoured) {
  route <- "A"
  cat("ROUTE A. One simulation, two batches, per-batch lib.loc. Splatter\n")
  cat("draws each batch's cells independently, so the biology is shared by\n")
  cat("construction and no pairing is invented. Use this.\n")
} else if (means_identical && de_identical && groups_identical && pairing_free) {
  route <- "B"
  cat("ROUTE B. Two single-batch calls sharing seed and biology. The gene\n")
  cat("means and the group DE factors agree and no per-cell pairing appeared,\n")
  cat("so the depth is the only difference between the sides.\n")
} else if (means_identical && de_identical && !pairing_free) {
  route <- "B_PAIRED"
  cat("ROUTE B IS UNSAFE. The biology is shared, but the shared seed also\n")
  cat("paired the sides cell by cell: a source cell's nearest target is its\n")
  cat("own index far more often than chance. That hands the transport a\n")
  cat("trivially correct answer unrelated to the biology under test. Do not\n")
  cat("generate on this route.\n")
} else {
  cat("NEITHER ROUTE IS SAFE. Two calls do not share their biology, so the\n")
  cat("sides would differ biologically and the benchmark's ground truth\n")
  cat("('every population is on both sides') would be an assumption rather\n")
  cat("than a construction.\n")
}

if (!is.null(out)) {
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  record <- c(route_a, list(
    splatter_version = as.character(packageVersion("splatter")),
    r_version = R.version.string,
    per_batch_lib_loc_kept = vector_kept,
    per_batch_lib_loc_honoured = route_a_honoured,
    two_call_gene_means_identical = means_identical,
    two_call_group_de_identical = de_identical,
    two_call_group_de_worst_abs_diff = de_worst,
    two_call_group_labels_identical = groups_identical,
    two_call_self_nearest_fraction = self_nearest,
    two_call_self_rank_mean = mean(self_rank),
    two_call_pairing_free = pairing_free,
    different_seed_self_nearest_fraction = self_nearest_o,
    different_seed_gene_means_identical = other_means_identical,
    asked_depth_ratio = DEPTH_RATIO,
    realised_depth_ratio_two_call = as.numeric(median(d_low) / median(d_high)),
    detection_rate_matched = as.numeric(median(f_high) / G),
    detection_rate_low_depth = as.numeric(median(f_low) / G),
    detection_rate_low_depth_dropout = as.numeric(median(f_drop) / G),
    dropout_type = as.character(getParam(drop_params, "dropout.type")),
    dropout_mid = as.numeric(getParam(drop_params, "dropout.mid")),
    dropout_shape = as.numeric(getParam(drop_params, "dropout.shape")),
    dropout_composes = l2_composes,
    route = route
  ))
  writeLines(jsonlite::toJSON(record, auto_unbox = TRUE, pretty = TRUE,
                              null = "null", na = "null"),
             file.path(out, "splatter_library_probe.json"))
  cat(sprintf("\nwrote %s\n", file.path(out, "splatter_library_probe.json")))
}
