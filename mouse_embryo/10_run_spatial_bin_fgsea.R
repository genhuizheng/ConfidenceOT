#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fgsea))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: 10_run_spatial_bin_fgsea.R <deg_root> <mouse_pathways.gmt> <output_root>")
}

deg_root <- normalizePath(args[[1]], mustWork = TRUE)
gmt_path <- normalizePath(args[[2]], mustWork = TRUE)
output_root <- args[[3]]
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)
pathways <- gmtPathways(gmt_path)
summaries <- list()

for (candidate_root in list.dirs(deg_root, recursive = FALSE, full.names = TRUE)) {
  rank_path <- file.path(candidate_root, "gsea_rank.rnk")
  if (!file.exists(rank_path)) next
  candidate <- basename(candidate_root)
  ranks_table <- read.delim(rank_path, header = FALSE, col.names = c("gene", "score"))
  ranks <- ranks_table$score
  names(ranks) <- ranks_table$gene
  ranks <- sort(ranks[is.finite(ranks)], decreasing = TRUE)
  result <- as.data.frame(fgseaMultilevel(
    pathways = pathways, stats = ranks, minSize = 10, maxSize = 500, eps = 0
  ))
  result$leadingEdge <- vapply(result$leadingEdge, paste, collapse = ";", FUN.VALUE = character(1))
  result <- result[order(result$padj, -abs(result$NES)), ]
  destination <- file.path(output_root, candidate)
  dir.create(destination, recursive = TRUE, showWarnings = FALSE)
  write.csv(result, file.path(destination, "fgsea_results.csv"), row.names = FALSE)
  summaries[[length(summaries) + 1]] <- data.frame(
    candidate = candidate,
    tested_pathway_n = nrow(result),
    pathway_fdr_005_n = sum(result$padj < 0.05, na.rm = TRUE)
  )
  message("Completed ", candidate)
}

if (length(summaries) == 0) stop("No candidate rank files were found.")
write.csv(do.call(rbind, summaries), file.path(output_root, "fgsea_summary.csv"), row.names = FALSE)
