#!/usr/bin/env Rscript
# Arm B step 2: Splatter parameters estimated from LTL331, and one simulation per estimate for the fidelity check.
#
#   benchmark/tacc/rscript.sh realdata_benchmark/splatter/22_estimate_ltl331.R MTX_ROOT OUT_ROOT
#
# For every MTX_ROOT/<name>/ (21_export_ltl331_mtx.py: the 8 time points and the pooled set): splatEstimate on the
# integer counts; the estimate is written as OUT_ROOT/<name>/params.rds and params.tsv (every parameter splatter
# reports); then splatSimulate with that estimate, one group, the same number of cells and genes, seed 20261031,
# written as OUT_ROOT/<name>/simulated/{matrix.mtx, genes.tsv, cells.tsv} for 23_fidelity_check.py. Nothing is tuned:
# the estimate is used as splatEstimate returns it. Dropout: splatEstimate fits dropout.mid / dropout.shape itself (splatEstDropout on normalised counts) but leaves
# dropout.type "none"; here the type is set to "experiment" so the simulation uses the estimated logistic, and both
# the estimate and the type are recorded.

suppressPackageStartupMessages({ library(splatter); library(Matrix) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: 22_estimate_ltl331.R MTX_ROOT OUT_ROOT")
mtx_root <- args[1]; out_root <- args[2]
if (dir.exists(out_root)) stop(paste(out_root, "exists; not overwritten"))
dir.create(out_root, recursive = TRUE)
cat("splatter", as.character(packageVersion("splatter")), "\n")

for (name in list.dirs(mtx_root, full.names = FALSE, recursive = FALSE)) {
  d <- file.path(mtx_root, name)
  counts <- as.matrix(readMM(file.path(d, "matrix.mtx")))
  storage.mode(counts) <- "integer"
  genes <- read.delim(file.path(d, "genes.tsv"), header = FALSE)$V1
  rownames(counts) <- make.unique(genes)
  colnames(counts) <- read.delim(file.path(d, "cells.tsv"), header = FALSE)$V1
  t0 <- Sys.time()
  params <- splatEstimate(counts)
  est <- setParam(params, "dropout.type", "experiment")
  o <- file.path(out_root, name); dir.create(o)
  saveRDS(est, file.path(o, "params.rds"))
  p <- getParams(est, c("mean.shape", "mean.rate", "lib.loc", "lib.scale", "lib.norm", "out.prob", "out.facLoc", "out.facScale",
                        "bcv.common", "bcv.df", "dropout.type", "dropout.mid", "dropout.shape", "nGenes", "batchCells"))
  write.table(data.frame(parameter = names(p), value = vapply(p, function(v) paste(format(v), collapse = ";"), "")),
              file.path(o, "params.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
  sim <- splatSimulate(est, nGenes = nrow(counts), batchCells = ncol(counts), seed = 20261031, verbose = FALSE)
  s <- file.path(o, "simulated"); dir.create(s)
  writeMM(as(counts(sim), "CsparseMatrix"), file.path(s, "matrix.mtx"))
  write.table(rownames(sim), file.path(s, "genes.tsv"), quote = FALSE, row.names = FALSE, col.names = FALSE)
  write.table(colnames(sim), file.path(s, "cells.tsv"), quote = FALSE, row.names = FALSE, col.names = FALSE)
  cat(name, ncol(counts), "cells,", nrow(counts), "genes,", format(Sys.time() - t0), "\n")
}
