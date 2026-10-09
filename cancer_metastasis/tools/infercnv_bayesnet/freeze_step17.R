# Run an inferCNV job script up to the BayesNet step (18) and freeze exactly
# what BayesNet would have received.
#
#   BAYESNET_RUN_SCRIPT=run_infercnv_no_inspect_saverds.R \
#   Rscript freeze_step17.R <the run script's own arguments>
#
# The run script is sourced unchanged. The only intervention is that
# infercnv::inferCNVBayesNet is replaced, in inferCNV's namespace, by a stub
# that saves its call arguments and stops the process cleanly. So steps 1-17 --
# filtering, normalisation, the background reference, denoising, Leiden
# subclusters and the HMM -- run exactly as in the unsharded job, once. With
# save_rds = TRUE in the run script they are also saved as inferCNV's own step
# objects, which the resume after the merge reloads.
#
# Writes into the run's out_dir:
#   bayesnet_frozen_inputs.rds  the infercnv_obj and every argument of the
#                               inferCNVBayesNet call except HMM_states (not
#                               read by sampling; inferCNV's step-17 object holds it)
#   bayesnet_freeze.tsv         md5 of the frozen inputs and of the step-17
#                               HMM files the regions are read from
#
# No set -u equivalent applies; this is R.

suppressPackageStartupMessages(library(infercnv))

run_script <- Sys.getenv("BAYESNET_RUN_SCRIPT")
if (!nzchar(run_script) || !file.exists(run_script)) {
    stop("BAYESNET_RUN_SCRIPT must name the inferCNV run script; got '", run_script, "'")
}

freeze_stub <- function(...) {
    call_args <- list(...)
    if (is.null(call_args$file_dir) || is.null(call_args$infercnv_obj)) {
        stop("inferCNVBayesNet was called without file_dir or infercnv_obj; cannot freeze")
    }
    out_dir <- call_args$file_dir
    call_args$HMM_states <- NULL
    frozen <- list(
        call_args       = call_args,
        infercnv_version = as.character(utils::packageVersion("infercnv")),
        run_script      = normalizePath(run_script),
        command_args    = commandArgs(trailingOnly = TRUE),
        frozen_at       = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
        slurm_job_id    = Sys.getenv("SLURM_JOB_ID", "local")
    )
    target <- file.path(out_dir, "bayesnet_frozen_inputs.rds")
    partial <- paste0(target, ".partial")
    saveRDS(frozen, partial, compress = FALSE)
    file.rename(partial, target)
    token <- call_args$resume_file_token
    hmm_files <- c(
        Sys.glob(file.path(out_dir, paste0("17_HMM_pred", token, ".cell_groupings"))),
        Sys.glob(file.path(out_dir, paste0("17_HMM_pred", token, ".pred_cnv_genes.dat"))),
        Sys.glob(file.path(out_dir, paste0("17_HMM_pred", token, ".infercnv_obj")))
    )
    files <- c(target, hmm_files)
    utils::write.table(data.frame(file = files, md5 = unname(tools::md5sum(files)),
                                  bytes = file.size(files)),
                       file.path(out_dir, "bayesnet_freeze.tsv"),
                       sep = "\t", quote = FALSE, row.names = FALSE)
    message("FROZEN at step 18: ", target)
    message("resume_file_token: ", token)
    quit(save = "no", status = 0)
}

utils::assignInNamespace("inferCNVBayesNet", freeze_stub, ns = "infercnv")
source(run_script, echo = FALSE)
stop("the run script finished without reaching inferCNVBayesNet: check HMM = TRUE and BayesMaxPNormal > 0")
