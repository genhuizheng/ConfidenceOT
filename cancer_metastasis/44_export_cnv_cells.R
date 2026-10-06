#!/usr/bin/env Rscript
# Export each cell's own inferred CNV profile for 02_run_pair.py
# --representation-source cnv or rna_cnv.
#
#   Rscript cancer_metastasis/44_export_cnv_cells.R PAIR_MANIFEST OUT_ROOT
#
# The patients are the manifest's (dataset_id, patient_id), the keys
# 02_run_pair.py looks a pair's profiles up by. For each,
# infercnv_r/runs_patient/<dataset_id>/<patient_id>/ is read: @expr.data of
# run.final.infercnv_obj, genes x cells, the expression-derived CNV signal
# inferCNV infers per cell, 1 being copy-neutral. It is an inferred profile, not
# a measured copy number. One patient's run centres primary and metastasis on
# one reference, so both sides are on one scale.
#
# The cells written are the run's observation cells, the columns inferCNV did
# not use as reference. The analysed malignant cells are among them -- every one
# of the 2026-10-02 run's was found there, none among the reference -- and the
# choice needs no cell-name join, which is where a barcode shared by two
# libraries would go wrong. A barcode that names two observation cells of one
# patient is left out, so a lookup cannot land on the wrong one; 02_run_pair.py
# stops on any analysed cell it cannot find.
#
# Written to OUT_ROOT/<dataset_id>/<patient_id>/:
#   cnv.f32    float32, cells x genes, one cell's genes after another
#   cells.tsv  cell (the matrix column), source_file, barcode (the h5ad obs_name)
#   genes.tsv  gene, chr, start, stop, in the matrix's row order
# and OUT_ROOT/export_summary.tsv. inferCNV is not rerun; nothing is recomputed.
# A patient without an object is reported, the others are still written, and
# the exit status is 1, so nothing waiting on this runs on a partial export. A
# complete export writes OUT_ROOT/EXPORT_COMPLETE, naming the manifest it covers.

suppressPackageStartupMessages(library(infercnv))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: 44_export_cnv_cells.R PAIR_MANIFEST OUT_ROOT")
manifest <- read.csv(args[1], colClasses = "character", check.names = FALSE)
units <- unique(manifest[, c("dataset_id", "patient_id")])
out_root <- args[2]
base <- Sys.getenv("CNV_BASE", "/scratch/10119/ghzheng/primary_metastatic_cancer")
runs <- file.path(base, "infercnv_r", "runs_patient")
cat(nrow(manifest), "pairs,", nrow(units), "patients in", args[1], "\n")

exported <- list()
missing <- character()
for (k in seq_len(nrow(units))) {
    d <- units$dataset_id[k]; p <- units$patient_id[k]
    unit <- file.path(runs, d, p)
    path <- file.path(unit, "infercnv_out", "run.final.infercnv_obj")
    if (!file.exists(path)) {
        cat(k, "of", nrow(units), d, p, "has no per-patient inferCNV object\n")
        missing <- c(missing, paste(d, p, sep = "/"))
        next
    }
    obj <- readRDS(path)
    x <- obj@expr.data
    meta <- read.delim(gzfile(file.path(unit, "cell_meta.tsv.gz")), colClasses = "character")
    meta <- meta[match(colnames(x), meta$cell), ]
    if (anyNA(meta$cell)) stop(d, "/", p, ": matrix columns missing from cell_meta.tsv.gz")
    observed <- sort(unique(unlist(obj@observation_grouped_cell_indices)))
    if (length(observed) == 0) stop(d, "/", p, ": no observation cells")
    shared <- duplicated(meta$barcode[observed]) | duplicated(meta$barcode[observed], fromLast = TRUE)
    keep <- observed[!shared]
    values <- as.vector(x[, keep, drop = FALSE])
    if (anyNA(values)) stop(d, "/", p, ": missing values in the matrix")
    dir <- file.path(out_root, d, p)
    dir.create(dir, recursive = TRUE, showWarnings = FALSE)
    con <- file(file.path(dir, "cnv.f32"), "wb")
    writeBin(values, con, size = 4)
    close(con)
    write.table(data.frame(cell = meta$cell[keep], source_file = meta$source_file[keep],
                           barcode = meta$barcode[keep]),
                file.path(dir, "cells.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
    go <- obj@gene_order[rownames(x), ]
    write.table(data.frame(gene = rownames(x), chr = go$chr, start = go$start, stop = go$stop),
                file.path(dir, "genes.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
    exported[[length(exported) + 1]] <- data.frame(
        dataset_id = d, patient_id = p, genes = nrow(x), cells = ncol(x),
        observation_cells = length(observed), shared_barcode_cells = sum(shared),
        exported = length(keep))
    cat(k, "of", nrow(units), d, p, nrow(x), "genes,", length(keep), "of", length(observed),
        "observation cells (", sum(shared), "with a shared barcode left out ) of", ncol(x), "\n")
    rm(obj, x, values); invisible(gc())
}
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)
res <- do.call(rbind, exported)
if (!is.null(res)) {
    write.table(res, file.path(out_root, "export_summary.tsv"), sep = "\t", quote = FALSE,
                row.names = FALSE)
    cat("\nexported", nrow(res), "patients,", sum(res$exported), "cells\n")
    print(res, row.names = FALSE)
}
if (length(missing)) {
    cat("\nno inferCNV object for", length(missing), "patient(s):", missing, "\n")
    quit(status = 1)
}
# Only a complete export says so; a rerun of the submission reuses it.
writeLines(normalizePath(args[1]), file.path(out_root, "EXPORT_COMPLETE"))
