# Merge region shards back into inferCNV's step-18 MCMC object.
#
#   Rscript merge.R --frozen <out_dir>/bayesnet_frozen_inputs.rds --shards-dir DIR
#
# Requires every region index of the plan exactly once across the shard files,
# then puts the samples in the original region order -- by index, never by
# file or completion order -- and finishes as infercnv::inferCNVBayesNet does
# after runMCMC: getProbabilities on the whole list, MCMC_inferCNV_obj.rds in
# the BayesNet output directory, and the "pre-filtering" normal-probability
# plot (which inferCNV draws only when plotting is on). The object is then saved
# under inferCNV's own step-18 file name in the run's out_dir, so rerunning the
# unchanged run script with resume reloads it as step 18 and carries on with
# step 19 -- BayesMaxPNormal filtering, removeCNV and reassignCNV -- and every
# later step in inferCNV's own code.
#
# Writes DIR/merge_record.tsv: per shard, regions, host, job id, start, end,
# seconds.

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
source(file.path(script_dir, "bayesnet_common.R"))

flags <- parse_flags(commandArgs(trailingOnly = TRUE))
if (is.null(flags$frozen) || is.null(flags[["shards-dir"]])) {
    stop("usage: merge.R --frozen FILE --shards-dir DIR")
}
shards_dir <- flags[["shards-dir"]]
frozen <- load_frozen(flags$frozen)
plan <- utils::read.delim(file.path(shards_dir, "shard_plan.tsv"), stringsAsFactors = FALSE)
files <- sort(Sys.glob(file.path(shards_dir, "shard_[0-9][0-9][0-9].rds")))
expected <- sort(unique(plan$shard))
found <- as.integer(sub("^shard_([0-9]+)\\.rds$", "\\1", basename(files)))
missing <- setdiff(expected, found)
if (length(missing)) stop("missing shard files: ", paste(missing, collapse = ", "))

n <- nrow(plan)
mcmc <- vector("list", n)
seen <- integer(n)
record <- list()
frozen_md5 <- unname(tools::md5sum(flags$frozen))
for (path in files) {
    shard <- readRDS(path)
    if (!identical(shard$frozen_md5, frozen_md5)) stop(path, " was sampled from a different frozen input")
    planned <- sort(plan$region_index[plan$shard == shard$shard])
    if (!identical(sort(as.integer(shard$region_index)), as.integer(planned))) {
        stop(path, " does not hold exactly the regions planned for shard ", shard$shard)
    }
    for (j in seq_along(shard$region_index)) {
        i <- shard$region_index[j]
        if (!identical(shard$region[j], plan$region[i])) stop("region name mismatch at index ", i)
        mcmc[i] <- list(shard$samples[[j]])
        seen[i] <- seen[i] + 1L
    }
    record[[length(record) + 1]] <- data.frame(shard = shard$shard, regions = length(shard$region_index),
                                                host = shard$host, slurm_job_id = shard$slurm_job_id,
                                                started = shard$started, finished = shard$finished,
                                                seconds = shard$seconds, rng = shard$rng)
}
if (any(seen != 1L)) stop("regions not covered exactly once: ", paste(which(seen != 1L), collapse = ", "))
utils::write.table(do.call(rbind, record), file.path(shards_dir, "merge_record.tsv"),
                   sep = "\t", quote = FALSE, row.names = FALSE)

obj <- build_mcmc_object(frozen)
if (!identical(vapply(obj@cell_gene, function(x) as.character(x$cnv_regions), character(1)), plan$region)) {
    stop("the frozen object's regions do not match the shard plan")
}
futile.logger::flog.info(paste("Obtaining probabilities post-sampling (merged from", length(files), "shards)"))
obj <- infercnv:::getProbabilities(obj, mcmc)

a <- frozen$call_args
bayes_dir <- obj@args$out_dir
dir.create(bayes_dir, showWarnings = FALSE, recursive = TRUE)
saveRDS(obj, file = file.path(bayes_dir, "MCMC_inferCNV_obj.rds"))
if (isTRUE(obj@args$postMcmcMethod == "removeCNV") || isTRUE(obj@args$reassignCNVs)) {
    title <- sprintf(" (1 - Probabilities of Normal) Before Filtering")
    output_filename <- "infercnv.NormalProbabilities.PreFiltering"
} else {
    title <- sprintf(" 1 - Probabilities of Normal ")
    output_filename <- "infercnv.NormalProbabilities"
}
infercnv:::postProbNormal(obj, PNormal = NULL, title = title, output_filename = output_filename,
                          useRaster = a$useRaster)

step18 <- file.path(a$file_dir, sprintf("18_HMM_pred.Bayes_Net%s.mcmc_obj", a$resume_file_token))
saveRDS(obj, paste0(step18, ".partial"))
file.rename(paste0(step18, ".partial"), step18)
message("merged ", n, " regions from ", length(files), " shards into ", step18)
