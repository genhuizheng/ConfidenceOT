# One CNV tool on one unit (the infercnvpy unit's cells, genes and reference, exported by 01_export_r_inputs.py), written
# in the layout 02_run_pair.py --cnv-root reads. Calls as verified in CNV_TOOL_APIS.md; package defaults except where
# a setting is named below.
#
#   Rscript 02_run_cnv_tool.R TOOL INPUT_DIR OUT_DIR SPECIES THREADS
#     TOOL     infercnv | copykat | scevan | conicsmat
#     SPECIES  mouse (KP-Tracer: gene symbols, GENCODE vM23 positions) | human (Quinn: Ensembl ids, GENCODE v28 / v19)
#
# Reference cells = cells.tsv is_reference (KP: the authors' normal cells; Quinn: none).
#   infercnv   CreateInfercnvObject + run(cutoff = 0.1 [the documented 10x setting], HMM = FALSE, analysis_mode =
#              "samples", no_plot = TRUE, resume_mode = FALSE); ref_group_names = the ref_ groups, or NULL (mean of all
#              cells) without a reference. Profile: obj@expr.data (genes x cells, centred at 1).
#   copykat    copykat(id.type = "S" mouse / "E" human, genome = "mm10" / "hg20", norm.cell.names = reference cells;
#              without a reference cell.line = "yes" (the README's setting for data with no normal cells), plot.genes =
#              "FALSE"). Profile: $CNAmat without its chrom / chrompos / abspos columns (bins x cells). Cells that fail
#              its filters are absent.
#   scevan     pipelineCNA(norm_cell = reference cells with FIXED_NORMAL_CELLS = TRUE, or NULL, organism, SUBCLONES =
#              FALSE, plotTree = FALSE). Profile: <sample>_CNAmtx.RData, CNA_mtx_relat (with normals) or CNA_mtx
#              (genes x cells). Cells that fail its filters are absent.
#   conicsmat  normMat -> filterMatrix -> calcNormFactors -> plotAll(regions = whole chromosomes built from this unit's
#              own gene positions: build-consistent, and mouse chromosomes have no arms). Profile: cells x regions
#              posterior probabilities (regions with <= 100 genes are dropped by plotAll).
# Written: OUT_DIR/{cnv.f32 (float32 cells x features, observation cells only), cells.tsv (cell, barcode),
#          genes.tsv (feature), coverage.json, provenance.json, SUCCESS}; FAILED with the reason otherwise. A barcode that
#          names two observation cells is left out (as 44_ / 49_). NA in the profile stops the unit (no imputation).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) stop("usage: Rscript 02_run_cnv_tool.R TOOL INPUT_DIR OUT_DIR SPECIES THREADS")
tool <- args[1]; input <- args[2]; out <- args[3]; species <- args[4]; threads <- as.integer(args[5])
stopifnot(tool %in% c("infercnv", "copykat", "scevan", "conicsmat"), species %in% c("mouse", "human"))
if (file.exists(file.path(out, "SUCCESS"))) { cat("done", out, "\n"); quit(status = 0) }
dir.create(out, recursive = TRUE, showWarnings = FALSE)
work <- file.path(out, "work"); dir.create(work, showWarnings = FALSE)
fail <- function(msg) { writeLines(msg, file.path(out, "FAILED")); stop(msg) }
t0 <- Sys.time()

suppressPackageStartupMessages(library(Matrix))
counts <- as(readMM(file.path(input, "counts.mtx")), "CsparseMatrix")
genes <- read.delim(file.path(input, "genes.tsv"), stringsAsFactors = FALSE)
cells <- read.delim(file.path(input, "cells.tsv"), stringsAsFactors = FALSE)
cells$is_reference <- as.logical(as.character(cells$is_reference))
dimnames(counts) <- list(genes$gene, cells$cell)
ref <- cells$cell[cells$is_reference]
obs <- cells$cell[!cells$is_reference]
cat(tool, input, "cells", ncol(counts), "reference", length(ref), "genes", nrow(counts), "\n")

profile <- NULL; call <- list()
if (tool == "infercnv") {
  suppressPackageStartupMessages(library(infercnv))
  go <- data.frame(chr = genes$chr, start = genes$start, stop = genes$stop, row.names = genes$gene)
  ann <- data.frame(group = cells$group, row.names = cells$cell)
  refs <- if (length(ref)) unique(cells$group[cells$is_reference]) else NULL
  obj <- CreateInfercnvObject(raw_counts_matrix = counts, annotations_file = ann, gene_order_file = go,
                              ref_group_names = refs)
  call <- list(cutoff = 0.1, HMM = FALSE, analysis_mode = "samples", ref_group_names = refs)
  obj <- infercnv::run(obj, cutoff = 0.1, out_dir = work, HMM = FALSE, analysis_mode = "samples",
                       no_plot = TRUE, resume_mode = FALSE, num_threads = threads)
  profile <- t(obj@expr.data)                                      # cells x genes
} else if (tool == "copykat") {
  suppressPackageStartupMessages(library(copykat))
  old <- setwd(work)
  idt <- if (species == "mouse") "S" else "E"; gen <- if (species == "mouse") "mm10" else "hg20"
  cl <- if (length(ref)) "no" else "yes"
  call <- list(id.type = idt, genome = gen, cell.line = cl, norm.cell.names = if (length(ref)) "reference cells" else "")
  ck <- copykat(rawmat = as.matrix(counts), id.type = idt, genome = gen, cell.line = cl,
                norm.cell.names = if (length(ref)) ref else "", plot.genes = "FALSE", sam.name = "unit", n.cores = threads)
  setwd(old)
  m <- ck$CNAmat
  m <- m[, !(colnames(m) %in% c("chrom", "chrompos", "abspos")), drop = FALSE]
  key <- setNames(cells$cell, make.names(cells$cell))              # copykat may sanitise names
  cn <- ifelse(colnames(m) %in% cells$cell, colnames(m), key[colnames(m)])
  if (anyNA(cn)) fail(sprintf("copykat: %d output columns match no cell", sum(is.na(cn))))
  profile <- t(as.matrix(m)); rownames(profile) <- cn
} else if (tool == "scevan") {
  suppressPackageStartupMessages(library(SCEVAN))
  old <- setwd(work)
  call <- list(organism = species, FIXED_NORMAL_CELLS = length(ref) > 0, SUBCLONES = FALSE, plotTree = FALSE)
  res <- pipelineCNA(as.matrix(counts), sample = "unit", par_cores = threads,
                     norm_cell = if (length(ref)) ref else NULL, FIXED_NORMAL_CELLS = length(ref) > 0,
                     SUBCLONES = FALSE, plotTree = FALSE, organism = species)
  setwd(old)
  f <- list.files(work, pattern = "_CNAmtx\\.RData$", recursive = TRUE, full.names = TRUE)
  if (length(f) != 1) fail(sprintf("scevan: %d _CNAmtx.RData files", length(f)))
  e <- new.env(); load(f, envir = e)
  nm <- intersect(c("CNA_mtx_relat", "CNA_mtx"), ls(e))
  if (!length(nm)) fail(paste("scevan: no CNA matrix object in", f, ":", paste(ls(e), collapse = ",")))
  call$matrix_object <- nm[1]
  profile <- t(as.matrix(get(nm[1], envir = e)))
} else {
  suppressPackageStartupMessages(library(CONICSmat))
  chrn <- sub("^chr", "", genes$chr); chrn[chrn == "X"] <- "23"; chrn[chrn == "Y"] <- "24"; chrn[chrn %in% c("M", "MT")] <- "0"
  gp <- data.frame(ensembl_gene_id = genes$gene, hgnc_symbol = genes$gene, chromosome_name = suppressWarnings(as.integer(chrn)),
                   start_position = genes$start, end_position = genes$stop, stringsAsFactors = FALSE)
  gp <- gp[!is.na(gp$chromosome_name), ]
  auto <- if (species == "mouse") 1:19 else 1:22
  reg <- do.call(rbind, lapply(auto, function(k) { s <- gp[gp$chromosome_name == k, ]
    data.frame(Chrom = k, Start = min(s$start_position), End = max(s$end_position)) }))
  reg$Length <- reg$End - reg$Start; rownames(reg) <- paste0("chr", reg$Chrom)
  expr <- normMat(as.matrix(counts[gp$ensembl_gene_id, ]))
  expr <- filterMatrix(expr, gp$hgnc_symbol)
  nf <- calcNormFactors(expr)
  call <- list(regions = "whole autosomes from this unit's gene positions", n_regions = nrow(reg))
  old <- setwd(work)
  l <- plotAll(expr, nf, reg, gp, file.path(work, "conics"))
  setwd(old)
  profile <- as.matrix(l)                                          # cells x regions
}

if (is.null(profile) || !nrow(profile)) fail(paste(tool, "returned no profile"))
present <- obs[obs %in% rownames(profile)]
bc <- cells$barcode[match(present, cells$cell)]
dup <- bc %in% bc[duplicated(bc)]
keep <- present[!dup]
P <- profile[keep, , drop = FALSE]
if (anyNA(P)) fail(sprintf("%s: %d NA values in the profile of %d cells", tool, sum(is.na(P)), sum(rowSums(is.na(P)) > 0)))
con <- file(file.path(out, "cnv.f32"), "wb"); writeBin(as.vector(t(P)), con, size = 4); close(con)
write.table(data.frame(cell = keep, barcode = cells$barcode[match(keep, cells$cell)]), file.path(out, "cells.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write.table(data.frame(feature = colnames(P)), file.path(out, "genes.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cov <- list(observation_cells = length(obs), in_profile = length(present), duplicated_barcodes = sum(dup),
            exported = length(keep), dropped_by_tool = length(obs) - length(present), features = ncol(P),
            reference_cells = length(ref))
ver <- tryCatch(as.character(packageVersion(c(infercnv = "infercnv", copykat = "copykat", scevan = "SCEVAN",
                                              conicsmat = "CONICSmat")[[tool]])), error = function(e) NA)
jsonlite::write_json(cov, file.path(out, "coverage.json"), auto_unbox = TRUE, pretty = TRUE)
jsonlite::write_json(list(tool = tool, version = ver, species = species, input = input, call = call, threads = threads,
                          seconds = as.numeric(difftime(Sys.time(), t0, units = "secs"))),
                     file.path(out, "provenance.json"), auto_unbox = TRUE, pretty = TRUE)
writeLines("ok", file.path(out, "SUCCESS"))
cat(tool, "exported", length(keep), "of", length(obs), "observation cells,", ncol(P), "features\n")
