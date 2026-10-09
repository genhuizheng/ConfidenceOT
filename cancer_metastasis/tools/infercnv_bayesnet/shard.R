# Plan and run region-level shards of inferCNV's BayesNet step.
#
#   Rscript shard.R --frozen <out_dir>/bayesnet_frozen_inputs.rds --plan --n-shards N --shards-dir DIR
#   Rscript shard.R --frozen <...> --shard K --shards-dir DIR
#
# The unit is one CNV region, as inferCNV defines it: one JAGS model per region
# over that region's genes and cells, sharing only the fixed state means and
# precisions (mu, sig) computed once from the hidden spike-in. A region is never
# split. Every shard rebuilds the MCMC object from the same frozen inputs and
# samples its regions with inferCNV's own run_gibb_sampling, one at a time in a
# single process (no fork), so a shard cannot lose a fork's result.
#
# --plan assigns whole regions to shards by cost (genes x cells, largest first,
# each to the least-loaded shard) and writes DIR/shard_plan.tsv: region index,
# name, state, genes, cells, the cell groups the region covers, and its shard.
# The assignment only schedules; the merge orders regions by index.
#
# --shard K samples the regions planned for shard K and writes
# DIR/shard_K.rds (atomically, via a .partial file): their indices, names and
# coda samples, plus host, job id and timing. An existing complete shard file is
# left alone, so a resubmission only reruns missing shards. No RNG seed is set:
# sampling matches inferCNV's own behaviour, which sets none.

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
source(file.path(script_dir, "bayesnet_common.R"))

flags <- parse_flags(commandArgs(trailingOnly = TRUE))
if (is.null(flags$frozen) || is.null(flags[["shards-dir"]])) {
    stop("usage: shard.R --frozen FILE --shards-dir DIR (--plan --n-shards N | --shard K)")
}
shards_dir <- flags[["shards-dir"]]
dir.create(shards_dir, showWarnings = FALSE, recursive = TRUE)
frozen <- load_frozen(flags$frozen)
plan_path <- file.path(shards_dir, "shard_plan.tsv")

if (isTRUE(flags$plan)) {
    n_shards <- as.integer(flags[["n-shards"]])
    if (is.na(n_shards) || n_shards < 1) stop("--plan needs --n-shards N >= 1")
    obj <- build_mcmc_object(frozen, cores = 1)
    regions <- region_table(obj)
    # The cell groups each region covers, read from the same step-17 file
    # initializeObject reads, and the annotation group each belongs to (the part
    # of inferCNV's cell_group_name before the first dot).
    token <- frozen$call_args$resume_file_token
    pred_path <- Sys.glob(file.path(frozen$call_args$file_dir, paste0("17_HMM_pred", token, ".pred_cnv_genes.dat")))
    pred <- utils::read.table(pred_path, header = TRUE, check.names = FALSE, sep = "\t", stringsAsFactors = FALSE)
    covered <- tapply(pred$cell_group_name, pred$gene_region_name, function(x) paste(sort(unique(x)), collapse = "|"))
    annotation <- tapply(pred$cell_group_name, pred$gene_region_name,
                         function(x) paste(sort(unique(sub("\\..*$", "", x))), collapse = "|"))
    regions$cell_groups <- unname(covered[regions$region])
    regions$annotation_groups <- unname(annotation[regions$region])
    regions$annotation_pure <- !grepl("\\|", regions$annotation_groups)
    regions$cost <- as.numeric(regions$n_genes) * as.numeric(regions$n_cells)
    load <- numeric(n_shards)
    regions$shard <- NA_integer_
    for (i in order(-regions$cost, regions$region_index)) {
        k <- which.min(load)
        regions$shard[i] <- k - 1L
        load[k] <- load[k] + regions$cost[i]
    }
    utils::write.table(regions, paste0(plan_path, ".partial"), sep = "\t", quote = FALSE, row.names = FALSE)
    file.rename(paste0(plan_path, ".partial"), plan_path)
    summary <- data.frame(shard = seq_len(n_shards) - 1L,
                          regions = as.integer(table(factor(regions$shard, levels = seq_len(n_shards) - 1L))),
                          cells_x_genes = load)
    message("regions: ", nrow(regions), "; empty (no cells): ", sum(regions$n_cells == 0),
            "; annotation-pure: ", sum(regions$annotation_pure), " of ", nrow(regions))
    print(summary, row.names = FALSE)
    quit(save = "no", status = 0)
}

k <- as.integer(flags$shard)
if (is.na(k)) stop("give --plan or --shard K")
if (!file.exists(plan_path)) stop("no shard plan at ", plan_path, "; run --plan first")
plan <- utils::read.delim(plan_path, stringsAsFactors = FALSE)
mine <- sort(plan$region_index[plan$shard == k])
target <- file.path(shards_dir, sprintf("shard_%03d.rds", k))
if (file.exists(target)) {
    message("shard ", k, " already complete: ", target)
    quit(save = "no", status = 0)
}
obj <- build_mcmc_object(frozen, cores = 1)
if (length(obj@cell_gene) != nrow(plan) ||
    !identical(vapply(obj@cell_gene, function(x) as.character(x$cnv_regions), character(1)), plan$region)) {
    stop("the frozen object's regions do not match the shard plan")
}
started <- Sys.time()
message("shard ", k, ": ", length(mine), " regions, ", sum(plan$cost[plan$shard == k]), " cells x genes")
samples <- vector("list", length(mine))
for (j in seq_along(mine)) {
    samples[[j]] <- sample_region(obj, mine[j])
    if (j %% 50 == 0 || j == length(mine)) {
        message(format(Sys.time(), "%Y-%m-%dT%H:%M:%S"), "  shard ", k, ": ", j, " of ", length(mine))
    }
}
result <- list(
    shard = k,
    region_index = mine,
    region = plan$region[match(mine, plan$region_index)],
    samples = samples,
    started = format(started, "%Y-%m-%dT%H:%M:%S"),
    finished = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
    seconds = as.numeric(difftime(Sys.time(), started, units = "secs")),
    host = Sys.info()[["nodename"]],
    slurm_job_id = Sys.getenv("SLURM_JOB_ID", "local"),
    rng = "not set (as inferCNV)",
    frozen_md5 = unname(tools::md5sum(flags$frozen))
)
partial <- paste0(target, ".partial")
saveRDS(result, partial)
file.rename(partial, target)
message("shard ", k, " done in ", round(result$seconds), " s: ", target)
