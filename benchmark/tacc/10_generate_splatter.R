#!/usr/bin/env Rscript
# Splatter generation for one technical condition of the benchmark.
#
# One call produces one (replicate, technical level): a single simulation
# carrying both sides as two batches, with the batch effect switched off so
# the only thing that can differ between them is the observation setting.
# If the batch effect were left on, the two sides would differ biologically
# and the ground truth "every population is on both sides" would not hold.
#
# **Each level is its own realization.** The three levels draw independently
# from the same biological parameters, with a seed that includes the level, so
# no two levels share realized counts. The earlier version derived L1 and L2
# from one draw by transforming it, which made a difference between levels
# partly a difference in the same cells rather than in the measurement, and
# left L2 defined as "L1 plus more dropout" -- a chain, not a condition.
#
# What varies between levels is the target side's observation setting and
# nothing else. Dropout is Splatter's own, applied per batch, so it belongs to
# the simulation rather than to a post-processing step. The depth mismatch
# cannot be expressed here -- Splatter has one global library size and no
# per-batch override -- so it stays downstream, as the one post-hoc operation,
# and it is applied to this level's own counts.
#
# Writes Matrix Market rather than h5ad: assembling an AnnData needs
# zellkonverter or anndata in R, and neither is installed on this account. The
# Python step reads these three files and writes the h5ad the pipeline wants.
#
# Usage:
#   Rscript 10_generate_splatter.R --out DIR --n 1000 --replicate 1 \
#       --level L0_matched --seed 20260925 \
#       [--groups 5] [--dropout-mid-target -1.0] [--estimate-from counts.mtx]

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
# Taken verbatim. The caller composes it from the replicate and the level so
# that what generation.json records is exactly what was used, rather than a
# base that has to be recombined by hand to be checked.
seed <- as.integer(value_of("--seed", "20260925"))
estimate_from <- value_of("--estimate-from", NULL)
dropout_mid_target <- value_of("--dropout-mid-target", NULL)
dropout_mid_source <- as.numeric(value_of("--dropout-mid-source", "-10"))
dropout_shape <- as.numeric(value_of("--dropout-shape", "-1"))

dir.create(out, recursive = TRUE, showWarnings = FALSE)

# Parameters come from a real matrix when one is given. Without it they are
# Splatter's published defaults, which is a weaker claim and is recorded as
# such rather than presented as an estimate.
provenance <- list()
if (!is.null(estimate_from) && file.exists(estimate_from)) {
  real <- as.matrix(readMM(estimate_from))
  params <- splatEstimate(real)
  provenance$parameter_source <- "splatEstimate"
  provenance$estimated_from <- estimate_from
} else {
  params <- newSplatParams()
  provenance$parameter_source <- "splatter defaults"
  provenance$estimated_from <- NA
}

# Everything set here is a benchmark control, and every one of them has a
# reason that belongs in the methods rather than in a comment alone.
params <- setParams(
  params,
  batchCells     = c(n_cells, n_cells),  # two sides, same size
  batch.facLoc   = 0,                    # no batch effect: the sides must
  batch.facScale = 0,                    #   differ only by measurement
  group.prob     = rep(1 / groups, groups),
  seed           = seed
)

# Dropout, per batch, so that it is the simulation's and not a mask applied
# afterwards. A very low midpoint with a negative shape puts the logistic
# essentially at zero, which is how the source side is left untouched while
# the target loses detection.
if (is.null(dropout_mid_target)) {
  params <- setParams(params, dropout.type = "none")
  dropout_record <- list(dropout_type = "none",
                         dropout_mid_source = NA,
                         dropout_mid_target = NA,
                         dropout_shape = NA)
} else {
  # Two calls, in this order. setParam validates dropout.type against the
  # length of dropout.mid and dropout.shape, so setting the type first is
  # refused: "set dropout.mid and dropout.shape first".
  params <- setParams(
    params,
    dropout.mid   = c(dropout_mid_source, as.numeric(dropout_mid_target)),
    dropout.shape = c(dropout_shape, dropout_shape)
  )
  params <- setParams(params, dropout.type = "batch")
  dropout_record <- list(dropout_type = "batch",
                         dropout_mid_source = dropout_mid_source,
                         dropout_mid_target = as.numeric(dropout_mid_target),
                         dropout_shape = dropout_shape)
}

sim <- splatSimulate(params, method = "groups", verbose = FALSE)

counts <- counts(sim)
cells <- data.frame(
  cell_id  = colnames(sim),
  batch    = as.character(colData(sim)$Batch),
  group    = as.character(colData(sim)$Group),
  stringsAsFactors = FALSE
)
genes <- data.frame(
  gene_id   = rownames(sim),
  gene_mean = as.numeric(rowData(sim)$GeneMean),
  stringsAsFactors = FALSE
)

writeMM(as(counts, "CsparseMatrix"), file.path(out, "counts.mtx"))
write.csv(cells, file.path(out, "cells.csv"), row.names = FALSE)
write.csv(genes, file.path(out, "genes.csv"), row.names = FALSE)

record <- c(
  provenance,
  dropout_record,
  list(
    n_cells_per_side = n_cells,
    groups = groups,
    replicate = replicate,
    technical_level = level,
    seed = seed,
    batch_fac_loc = 0,
    batch_fac_scale = 0,
    independent_realization = TRUE,
    splatter_version = as.character(packageVersion("splatter")),
    r_version = R.version.string,
    de_prob = getParam(params, "de.prob"),
    de_fac_loc = getParam(params, "de.facLoc"),
    de_fac_scale = getParam(params, "de.facScale"),
    lib_loc = getParam(params, "lib.loc"),
    lib_scale = getParam(params, "lib.scale"),
    bcv_common = getParam(params, "bcv.common"),
    bcv_df = getParam(params, "bcv.df"),
    mean_rate = getParam(params, "mean.rate"),
    mean_shape = getParam(params, "mean.shape")
  )
)
writeLines(jsonlite::toJSON(record, auto_unbox = TRUE, pretty = TRUE,
                            null = "null", na = "null"),
           file.path(out, "generation.json"))
writeLines("SUCCESS", file.path(out, "GENERATION_DONE"))
cat(sprintf("wrote %s (%d cells per side, %d groups, level %s, seed %d)\n",
            out, n_cells, groups, level, seed))
