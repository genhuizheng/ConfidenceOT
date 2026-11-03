#!/usr/bin/env Rscript
# The installed Splatter's default parameters and the code that applies batch factors and dropout (read only).
#
#   benchmark/tacc/rscript.sh realdata_benchmark/splatter_n5000/28_splatter_defaults.R --out DIR
#
# Writes DIR/SPLATTER_DEFAULTS.tsv (every scalar / vector parameter of newSplatParams(), as text) and
# DIR/SPLATTER_BATCH_CODE.txt (package version and the bodies of getLNormFactors, splatSimBatchEffects,
# splatSimBatchCellMeans, splatSimGroupCellMeans, splatSimDropout, and the order of steps in splatSimulate).

suppressPackageStartupMessages(library(splatter))
args <- commandArgs(trailingOnly = TRUE)
out <- args[which(args == "--out")[1] + 1]
if (is.na(out)) stop("--out is required")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
p <- newSplatParams()
slots <- methods::slotNames(p)
vals <- vapply(slots, function(s) paste(format(methods::slot(p, s)), collapse = ","), character(1))
write.table(data.frame(parameter = slots, default = vals), file.path(out, "SPLATTER_DEFAULTS.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
ns <- asNamespace("splatter")
con <- file(file.path(out, "SPLATTER_BATCH_CODE.txt"), "w")
writeLines(sprintf("splatter %s, %s", as.character(packageVersion("splatter")), R.version.string), con)
for (f in c("getLNormFactors", "splatSimBatchEffects", "splatSimBatchCellMeans", "splatSimGroupCellMeans", "splatSimDropout", "splatSimulate")) {
  writeLines(c("", paste("=====", f), deparse(get(f, envir = ns))), con)
}
close(con)
cat("->", out, "\n")
