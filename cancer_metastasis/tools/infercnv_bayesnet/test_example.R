# Test the sharded BayesNet workflow end to end on inferCNV's bundled example.
#
#   Rscript test_example.R <work_dir> [n_shards]
#
# The example (oligodendroglioma, 184 cells, two non-malignant reference
# groups) runs with the prostate settings: i6 HMM, subclusters, cluster by
# groups, denoise, no plots, inspect_subclusters FALSE, save_rds TRUE,
# BayesMaxPNormal 0.5. Then:
#   1. freeze_step17.R stops it at step 18 and freezes the BayesNet inputs;
#   2. shard.R plans n shards and samples each in its own process;
#   3. merge.R merges them into the step-18 object;
#   4. the unchanged run script runs again, and must resume from step 18;
#   5. the real infercnv::inferCNVBayesNet is called once on the same frozen
#      inputs, unsharded, as the comparison.
# Checks:
#   - the resume log says "Using backup MCMC from step 18" and never "STEP 18: Run Bayesian";
#   - the resumed run finishes with run.final.infercnv_obj;
#   - the merged object has the same regions, in the same order, as the unsharded one;
#   - per-region P(normal) agrees with the unsharded run within Monte Carlo error
#     (no RNG seed is set by inferCNV, so the two are not bit-identical; a
#     region-order error would show as disagreement far beyond that);
#   - the removeCNV decisions at 0.5 agree except for regions within 0.1 of
#     the threshold.
# Prints PASS or FAIL and exits accordingly.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: test_example.R <work_dir> [n_shards]")
work <- normalizePath(args[1], mustWork = FALSE)
n_shards <- if (length(args) > 1) as.integer(args[2]) else 3L
script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
dir.create(work, recursive = TRUE, showWarnings = FALSE)
suppressPackageStartupMessages(library(infercnv))

run_script <- file.path(work, "run_example.R")
writeLines(c(
    "suppressPackageStartupMessages(library(infercnv))",
    "out_dir <- Sys.getenv('EXAMPLE_OUT')",
    "obj <- infercnv::CreateInfercnvObject(",
    "    raw_counts_matrix = system.file('extdata', 'oligodendroglioma_expression_downsampled.counts.matrix.gz', package = 'infercnv'),",
    "    annotations_file  = system.file('extdata', 'oligodendroglioma_annotations_downsampled.txt', package = 'infercnv'),",
    "    delim = '\\t',",
    "    gene_order_file   = system.file('extdata', 'gencode_downsampled.EXAMPLE_ONLY_DONT_REUSE.txt', package = 'infercnv'),",
    "    ref_group_names   = c('Microglia/Macrophage', 'Oligodendrocytes (non-malignant)'))",
    "obj <- infercnv::run(obj, cutoff = 1, out_dir = out_dir, cluster_by_groups = TRUE,",
    "    analysis_mode = 'subclusters', denoise = TRUE, HMM = TRUE, HMM_type = 'i6',",
    "    num_threads = 1, no_plot = TRUE, no_prelim_plot = TRUE, write_expr_matrix = FALSE,",
    "    inspect_subclusters = FALSE, save_rds = TRUE, save_final_rds = TRUE)"
), run_script)

rscript <- file.path(R.home("bin"), "Rscript")
run <- function(script, extra = character(), env = character(), log) {
    status <- system2(rscript, c(script, extra), env = env, stdout = log, stderr = log)
    message(basename(script), " ", paste(extra, collapse = " "), " -> exit ", status, " (log ", log, ")")
    status
}
out <- file.path(work, "sharded")
unlink(out, recursive = TRUE)
dir.create(out)
env_out <- paste0("EXAMPLE_OUT=", out)
failures <- character()
check <- function(ok, what) {
    message(if (ok) "ok    " else "FAIL  ", what)
    if (!ok) failures <<- c(failures, what)
}

# 1. Freeze at step 18.
st <- run(file.path(script_dir, "freeze_step17.R"),
          env = c(env_out, paste0("BAYESNET_RUN_SCRIPT=", run_script)), log = file.path(work, "1_freeze.log"))
frozen <- file.path(out, "bayesnet_frozen_inputs.rds")
check(st == 0 && file.exists(frozen), "freeze wrote bayesnet_frozen_inputs.rds")
shards_dir <- file.path(work, "shards")
unlink(shards_dir, recursive = TRUE)

# 2. Plan and sample.
st <- run(file.path(script_dir, "shard.R"), c("--frozen", frozen, "--plan", "--n-shards", n_shards,
                                              "--shards-dir", shards_dir), log = file.path(work, "2_plan.log"))
check(st == 0 && file.exists(file.path(shards_dir, "shard_plan.tsv")), "plan written")
for (k in seq_len(n_shards) - 1L) {
    st <- run(file.path(script_dir, "shard.R"), c("--frozen", frozen, "--shard", k, "--shards-dir", shards_dir),
              log = file.path(work, sprintf("3_shard_%d.log", k)))
    check(st == 0, sprintf("shard %d sampled", k))
}

# 3. Merge.
st <- run(file.path(script_dir, "merge.R"), c("--frozen", frozen, "--shards-dir", shards_dir),
          log = file.path(work, "4_merge.log"))
step18 <- Sys.glob(file.path(out, "18_HMM_pred.Bayes_Net*.mcmc_obj"))
check(st == 0 && length(step18) == 1, "merge wrote the step-18 object")

# 4. Resume with the unchanged run script.
resume_log <- file.path(work, "5_resume.log")
st <- run(run_script, env = env_out, log = resume_log)
log_text <- readLines(resume_log)
check(st == 0, "resumed run exited 0")
check(any(grepl("Using backup MCMC from step 18", log_text)), "resume reloaded the merged step-18 object")
check(!any(grepl("STEP 18: Run Bayesian", log_text)), "resume did not rerun BayesNet")
check(any(grepl("STEP 19", log_text)), "resume ran step 19 (BayesMaxPNormal filtering)")
check(file.exists(file.path(out, "run.final.infercnv_obj")), "resumed run wrote run.final.infercnv_obj")

# 5. Unsharded BayesNet on the same frozen inputs, for comparison.
fz <- readRDS(frozen)
a <- fz$call_args
ref_dir <- file.path(work, "unsharded_bayesnet")
unlink(ref_dir, recursive = TRUE)
hmm_obj <- readRDS(Sys.glob(file.path(out, paste0("17_HMM_pred", a$resume_file_token, ".infercnv_obj"))))
reference <- infercnv::inferCNVBayesNet(infercnv_obj = a$infercnv_obj, HMM_states = hmm_obj@expr.data,
    file_dir = a$file_dir, no_plot = a$no_plot, postMcmcMethod = a$postMcmcMethod, out_dir = ref_dir,
    resume_file_token = a$resume_file_token, quietly = TRUE, CORES = 1, plotingProbs = a$plotingProbs,
    diagnostics = a$diagnostics, HMM_type = a$HMM_type, k_obs_groups = a$k_obs_groups,
    cluster_by_groups = a$cluster_by_groups, reassignCNVs = a$reassignCNVs, useRaster = a$useRaster)
merged <- readRDS(step18)
regions_of <- function(o) vapply(o@cell_gene, function(x) as.character(x$cnv_regions), character(1))
check(identical(regions_of(merged), regions_of(reference)), "merged regions identical and in the same order as unsharded")
check(identical(merged@mu, reference@mu) && identical(merged@sig, reference@sig), "state means and precisions identical")
check(identical(merged@options, a$infercnv_obj@options), "merged object carries the step-17 options")
p_normal <- function(o) vapply(o@cnv_probabilities, function(x) if (is.null(x)) NA_real_ else colMeans(x)[3], numeric(1))
pm <- p_normal(merged)
pr <- p_normal(reference)
both <- is.finite(pm) & is.finite(pr)
difference <- abs(pm - pr)[both]
message(sprintf("regions %d; P(normal) |merged - unsharded|: median %.4f, max %.4f; correlation %.4f",
                length(pm), stats::median(difference), max(difference), stats::cor(pm[both], pr[both])))
check(stats::cor(pm[both], pr[both]) > 0.95, "per-region P(normal) correlates > 0.95 with the unsharded run")
decided <- both & abs(pr - 0.5) > 0.1
check(all((pm[decided] > 0.5) == (pr[decided] > 0.5)), "removeCNV decisions agree away from the 0.5 threshold")
utils::write.table(data.frame(region = regions_of(merged), p_normal_merged = pm, p_normal_unsharded = pr),
                   file.path(work, "p_normal_comparison.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
if (length(failures)) {
    message("FAIL: ", paste(failures, collapse = "; "))
    quit(save = "no", status = 1)
}
message("PASS")
