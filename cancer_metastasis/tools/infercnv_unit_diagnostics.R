# Is an R inferCNV unit's reference close to diploid, and does it hold cells
# that look like the malignant ones?
#
#   Rscript cancer_metastasis/tools/infercnv_unit_diagnostics.R UNIT_DIR
#
# UNIT_DIR holds infercnv_out/run.final.infercnv_obj and cell_meta.tsv.gz
# (cell, source_file, is_reference, total_counts), as the collection's
# exporters write them. From the inferred single-cell CNV matrix (@expr.data),
# per cell, the two axes of Tirosh et al. 2016 and Puram et al. 2017:
#   signal       mean over genes of (x - 1)^2
#   correlation  correlation of (x - 1) with the mean (x - 1) profile of the
#                patient's malignant (observation) cells
# A reference built from diploid cells sits low on both. Reference cells at or
# above the malignant cells' 10th percentile on BOTH axes are counted as
# malignant-like, by site and file. Because the reference is what the matrix is
# centred on, its own signal is low partly by construction; what can show
# contamination is a reference subset that follows the malignant profile.
#
# Two things the pilot version did not handle. Denoising can leave a cell's
# profile flat -- the same value at every gene -- and a flat profile has no
# correlation with anything; such cells are counted and reported, they have no
# deviation from the reference at any gene, and they are not malignant-like.
# And signal is partly sequencing depth, since a shallow cell's profile is
# noisier, so it is also shown within depth quintiles of the pooled cells.
#
# The HMM-based altered fraction of the per-cell table is reported beside it,
# as sep and ratio in the form infercnv_r/summarise_runs.py uses.

suppressPackageStartupMessages(library(infercnv))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: infercnv_unit_diagnostics.R UNIT_DIR")
dir <- args[1]
obj <- readRDS(file.path(dir, "infercnv_out", "run.final.infercnv_obj"))
x <- obj@expr.data
rm(obj); invisible(gc())
meta <- read.delim(gzfile(file.path(dir, "cell_meta.tsv.gz")), stringsAsFactors = FALSE)
meta <- meta[match(colnames(x), meta$cell), ]
if (anyNA(meta$cell)) stop("matrix columns missing from cell_meta.tsv.gz")
cat("matrix:", nrow(x), "genes x", ncol(x), "cells\n")

site <- ifelse(grepl("__metastasis__", meta$source_file), "metastasis",
               ifelse(grepl("__primary__", meta$source_file), "primary", "other"))
role <- ifelse(meta$is_reference == 1, "reference", "malignant")
d <- x - 1
rm(x); invisible(gc())
signal <- colMeans(d^2)
consensus <- rowMeans(d[, role == "malignant", drop = FALSE])
correlation <- suppressWarnings(as.vector(cor(d, consensus)))
flat <- !is.finite(correlation)
cells <- data.frame(cell = meta$cell, role = role, site = site, file = meta$source_file,
                    total_counts = meta$total_counts, signal = signal, correlation = correlation,
                    flat = flat)
gz <- gzfile(file.path(dir, "pilot_per_cell_axes.tsv.gz"), "w")
write.table(cells, gz, sep = "\t", quote = FALSE, row.names = FALSE)
close(gz)

describe <- function(v) round(quantile(v, c(0.05, 0.5, 0.95, 0.99), na.rm = TRUE), 5)
groups <- expand.grid(s = c("primary", "metastasis"), r = c("malignant", "reference"),
                      stringsAsFactors = FALSE)
cat("\nsignal, by role and site (5%, 50%, 95%, 99%):\n")
for (k in seq_len(nrow(groups))) {
    v <- cells$signal[cells$role == groups$r[k] & cells$site == groups$s[k]]
    if (length(v)) cat(sprintf("  %-9s %-10s n=%6d  %s\n", groups$r[k], groups$s[k], length(v),
                               paste(describe(v), collapse = "  ")))
}
cat("\ncorrelation with the malignant consensus, by role and site (5%, 50%, 95%, 99%),",
    "over the cells whose profile is not flat:\n")
for (k in seq_len(nrow(groups))) {
    keep <- cells$role == groups$r[k] & cells$site == groups$s[k]
    if (!any(keep)) next
    v <- cells$correlation[keep & !cells$flat]
    cat(sprintf("  %-9s %-10s n=%6d  flat %6d (%.3f)  %s\n", groups$r[k], groups$s[k], sum(keep),
                sum(cells$flat[keep]), mean(cells$flat[keep]),
                if (length(v)) paste(describe(v), collapse = "  ") else "none"))
}

cat("\nmedian signal within depth quintiles of all cells (total_counts), cells in brackets:\n")
breaks <- unique(quantile(cells$total_counts, seq(0, 1, 0.2), na.rm = TRUE))
quintile <- cut(cells$total_counts, breaks, include.lowest = TRUE, dig.lab = 6)
medians <- tapply(cells$signal, list(quintile, cells$role), median)
counts <- table(quintile, cells$role)
for (q in rownames(medians)) {
    cat(sprintf("  %-22s malignant %.5f [%6d]   reference %.5f [%6d]\n", q,
                medians[q, "malignant"], counts[q, "malignant"],
                medians[q, "reference"], counts[q, "reference"]))
}

mal <- cells[cells$role == "malignant", ]
ref <- cells[cells$role == "reference", ]
cut_signal <- quantile(mal$signal, 0.10)
cut_correlation <- quantile(mal$correlation, 0.10, na.rm = TRUE)
ref$malignant_like <- ref$signal >= cut_signal & !ref$flat & ref$correlation >= cut_correlation
ref$malignant_like[is.na(ref$malignant_like)] <- FALSE
cat(sprintf("\nmalignant 10th percentiles: signal %.5f, correlation %.4f\n", cut_signal, cut_correlation))
cat(sprintf("reference cells at or above both: %d of %d (%.4f); %d reference cells are flat\n",
            sum(ref$malignant_like), nrow(ref), mean(ref$malignant_like), sum(ref$flat)))
print(aggregate(malignant_like ~ site + file, data = ref,
                FUN = function(v) c(n = length(v), malignant_like = sum(v), share = round(mean(v), 4))))
ref_95 <- c(quantile(ref$signal, 0.95), quantile(ref$correlation, 0.95, na.rm = TRUE))
above <- mal$signal >= ref_95[1] & !mal$flat & mal$correlation >= ref_95[2]
cat(sprintf("\nfor scale, malignant cells above the reference's 95th percentile on both axes: %.4f\n",
            mean(above %in% TRUE)))

per_cell <- file.path(dir, "per_cell_infercnv_r.tsv.gz")
if (file.exists(per_cell)) {
    t <- read.delim(gzfile(per_cell), stringsAsFactors = FALSE)
    altered <- rowMeans(as.matrix(t[, grep("^proportion_cnv_chr", names(t))]))
    is_ref <- t$is_reference == 1
    cat(sprintf("\nHMM altered fraction: reference mean %.4f, malignant mean %.4f, sep %.4f, ratio %.2f\n",
                mean(altered[is_ref]), mean(altered[!is_ref]),
                mean(altered[!is_ref]) - mean(altered[is_ref]),
                mean(altered[!is_ref]) / mean(altered[is_ref])))
    cat("for scale, over the 132 patient units of the collection: sep median 0.28, ratio median 7.87\n")
} else {
    cat("\nno per_cell_infercnv_r.tsv.gz, so no HMM altered fraction\n")
}
cat("\nwritten:", file.path(dir, "pilot_per_cell_axes.tsv.gz"), "\n")
