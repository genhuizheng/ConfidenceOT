#!/usr/bin/env Rscript
# Splatter generation for the technical-degradation benchmark.
#
# One call produces one replicate: a single simulation carrying both sides as
# two batches, with the batch effect switched off so the only thing that can
# differ between them is what the construction step does afterwards. If the
# batch effect were left on, the two sides would differ biologically and the
# ground truth "every population is on both sides" would not hold.
#
# Dropout is switched off here and applied per batch by the construction step,
# because the level ladder needs the source untouched while the target
# degrades. Splatter has no per-batch library size, so depth mismatch is not
# expressible here either; both live downstream.
#
# Writes Matrix Market rather than h5ad: assembling an AnnData needs
# zellkonverter or anndata in R, and neither is installed on this account. The
# Python step reads these three files and writes the h5ad the pipeline wants.
#
# Usage:
#   Rscript 10_generate_splatter.R --out DIR --n 1000 --replicate 1 \
#       [--groups 5] [--seed 20260925] [--estimate-from counts.mtx]

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
groups <- as.integer(value_of("--groups", "5"))
seed <- as.integer(value_of("--seed", "20260925")) + 1000L * replicate
estimate_from <- value_of("--estimate-from", NULL)

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
  dropout.type   = "none",               # applied per batch downstream
  seed           = seed
)

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
  list(
    n_cells_per_side = n_cells,
    groups = groups,
    replicate = replicate,
    seed = seed,
    batch_fac_loc = 0,
    batch_fac_scale = 0,
    dropout_type_at_generation = "none",
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
cat(sprintf("wrote %s (%d cells per side, %d groups, seed %d)\n",
            out, n_cells, groups, seed))
