# Shared by plan, shard and merge: rebuild inferCNV's MCMC object from the
# frozen inputs exactly as infercnv::inferCNVBayesNet (inferCNV 1.26.0,
# R/inferCNV_BayesNet.R) builds it before sampling -- the same args list, the
# same initializeObject (regions, cells and genes read from the step-17 HMM
# files) and the same MeanSD (state means and precisions from the hidden
# spike-in). Nothing here samples.

suppressPackageStartupMessages(library(infercnv))

load_frozen <- function(path) {
    frozen <- readRDS(path)
    version <- as.character(utils::packageVersion("infercnv"))
    if (!identical(frozen$infercnv_version, version)) {
        stop("frozen with inferCNV ", frozen$infercnv_version, " but this R has ", version)
    }
    frozen
}

build_mcmc_object <- function(frozen, cores = NULL) {
    a <- frozen$call_args
    model_file <- a$model_file
    if (is.null(model_file)) {
        model_file <- ifelse(a$HMM_type == "i6",
                             system.file("BUGS_Mixture_Model", package = "infercnv"),
                             system.file("BUGS_Mixture_Model_i3", package = "infercnv"))
    }
    plotingProbs <- if (is.null(a$plotingProbs)) TRUE else a$plotingProbs
    diagnostics <- if (is.null(a$diagnostics)) FALSE else a$diagnostics
    if (isTRUE(a$no_plot)) {
        plotingProbs <- FALSE
        diagnostics <- FALSE
    }
    args_parsed <- list("file_dir" = a$file_dir,
                        "model_file" = model_file,
                        "CORES" = if (is.null(cores)) a$CORES else cores,
                        "out_dir" = a$out_dir,
                        "resume_file_token" = a$resume_file_token,
                        "plotingProbs" = plotingProbs,
                        "postMcmcMethod" = a$postMcmcMethod,
                        "quietly" = if (is.null(a$quietly)) TRUE else a$quietly,
                        "BayesMaxPNormal" = 0,
                        "HMM_type" = a$HMM_type,
                        "k_obs_groups" = a$k_obs_groups,
                        "cluster_by_groups" = a$cluster_by_groups,
                        "reassignCNVs" = if (is.null(a$reassignCNVs)) TRUE else a$reassignCNVs,
                        "diagnostics" = diagnostics)
    obj <- methods::new("MCMC_inferCNV")
    obj <- infercnv:::initializeObject(obj, args_parsed, a$infercnv_obj)
    obj <- infercnv:::MeanSD(obj = obj, HMM_states = NULL, infercnv_obj = a$infercnv_obj)
    obj
}

region_table <- function(obj) {
    cg <- obj@cell_gene
    groups <- vapply(seq_along(cg), function(i) {
        cells <- cg[[i]]$Cells
        if (!length(cells)) return("")
        names <- unique(obj@group_id[cells])
        paste(sort(unique(names)), collapse = "|")
    }, character(1))
    data.frame(
        region_index = seq_along(cg),
        region = vapply(cg, function(x) as.character(x$cnv_regions), character(1)),
        state = vapply(cg, function(x) paste(x$State, collapse = "|"), character(1)),
        n_genes = vapply(cg, function(x) length(x$Genes), integer(1)),
        n_cells = vapply(cg, function(x) length(x$Cells), integer(1)),
        group_ids = groups,
        stringsAsFactors = FALSE
    )
}

# One region, exactly as nonParallel/withParallel's per-region body.
sample_region <- function(obj, i) {
    if (!(length(obj@cell_gene[[i]]$Cells) == 0)) {
        gene_exp <- obj@expr.data[obj@cell_gene[[i]]$Genes, obj@cell_gene[[i]]$Cells]
        return(infercnv:::run_gibb_sampling(gene_exp, obj))
    }
    list(NULL)
}

parse_flags <- function(args) {
    out <- list()
    i <- 1
    while (i <= length(args)) {
        key <- sub("^--", "", args[i])
        if (i == length(args) || startsWith(args[i + 1], "--")) {
            out[[key]] <- TRUE
            i <- i + 1
        } else {
            out[[key]] <- args[i + 1]
            i <- i + 2
        }
    }
    out
}
