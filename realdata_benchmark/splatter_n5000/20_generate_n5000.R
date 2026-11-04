#!/usr/bin/env Rscript
# Splatter N = 5,000 benchmark, cumulative levels L0-L5, built on the native generator
# (benchmark/tacc/10_generate_splatter_native.R, route D: same seeds, same route, same output files).
#
#   benchmark/tacc/rscript.sh realdata_benchmark/splatter_n5000/20_generate_n5000.R --out DIR --n 5000 \
#       --replicate R --level LEVEL --seed S [--param-set benchmark | splatter_default]
#
# --param-set (added 2026-11-03; default benchmark = the frozen L0-L5 of splatter_n5000_benchmark_20261101):
#   benchmark         the values written below: batch.facLoc 0, batch.facScale sqrt(0.5 D / 2), target-only dropout
#                     with dropout.mid 1.0 (--dropout-mid-target), dropout.shape -1.
#   splatter_default  every Splatter parameter at the installed package's newSplatParams() default, read at run time
#                     and recorded: a level with batch uses the default batch.facLoc / batch.facScale (0 / 0 when the
#                     level has no batch, so the factor is absent); a level with dropout uses dropout.type "experiment"
#                     with the default dropout.mid / dropout.shape on BOTH sides; lib.loc / lib.scale are the defaults.
#                     Kept as design factors: the target depth (lib.loc + log 0.5, L1), the Group1 perturbation
#                     (drawn from the default DE parameters, as below) and the L5 target proportions. L0 and L1 are the
#                     same simulation under either set (no batch, no dropout).
# --pairing (added 2026-11-03; default none = everything above unchanged; paired needs --param-set splatter_default):
#   paired            makes the levels of one replicate differ only by their own factors.
#                     (1) Batch: the default batch.facLoc / batch.facScale are used in EVERY level, so Splatter draws
#                     the batch factors (and so consumes the same random numbers) everywhere; a level without batch sets
#                     the drawn factors to 1 at the entry of splatSimBatchCellMeans (trace), before they touch the means.
#                     Without this, rnorm(sd = 0) draws nothing at scale 0 and the group DE factors that follow are a
#                     different draw (BATCH_EFFECT_AUDIT_20261103.md section 3).
#                     (2) Counts: Splatter's own BCV / Poisson / dropout draws (rchisq, rgamma, rpois, rbinom over the
#                     whole matrix) shift the random stream for every later entry whenever any mean changes. In paired
#                     mode the counts are re-sampled from Splatter's BaseCellMeans with the same model and parameters
#                     (bcv = (bcv.common + 1 / sqrt(mean)) x sqrt(bcv.df / chisq_gene); CellMeans ~ Gamma(1 / bcv^2,
#                     mean x bcv^2); TrueCounts ~ Poisson(CellMeans); dropout keep ~ Bernoulli(1 - logistic(log CellMeans,
#                     dropout.mid, dropout.shape))) by inverse CDF of common uniforms fixed per replicate and call
#                     (seed + 271829 deep, seed + 271830 shallow), drawn in that order whatever the level. An entry whose
#                     latent mean is unchanged gets exactly the same count; a changed mean moves its count monotonically.
#                     This sampler is ours (custom), not Splatter's code; sampler_check.csv compares it per gene with
#                     Splatter's own counts of the same call (means, zero fractions).
#
# N semantics, as the native benchmark: --n is cells PER SIDE (batchCells = c(N, N)); source = batch 1 of the deep
# call, target = batch 2 of the shallow call. N = 5,000 means 5,000 source + 5,000 target cells. Recorded.
#
# Levels. Each inherits every factor of the one before and adds exactly one (values fixed here, before any OT):
#   L0_matched                  matched depth, no dropout                                   (frozen, native)
#   L1_depth                    target library size x 0.5 (lib.loc + log 0.5)                (frozen, native)
#   L2_depth_dropout            L1 + Splatter dropout on the target (mid 1.0, shape -1)      (frozen, native)
#   L3_batch                    L2 + source / target batch effect: Splatter's own batch factors, batch.facLoc 0,
#                               batch.facScale = sqrt(0.5 * D / 2), so the per-gene log shift Batch2 / Batch1 has
#                               expected square 0.5 * D (D below)
#   L4_batch_perturb            L3 + target-side transcriptional perturbation of Group1 (shared in all_shared,
#                               population_lost, population_emerged; it keeps its label): gene factors with the group-DE
#                               distribution at probability 2 * 0.5 * de.prob, log factor +-N(de.facLoc, de.facScale),
#                               down with de.downProb, expected squared log shift 0.5 * D; applied to the latent cell
#                               means (BaseCellMeans at the entry of splatSimBCVMeans, by trace()), so BCV, counts and
#                               dropout follow as Splatter draws them
#   L5_batch_perturb_imbalance  L4 + target population proportions (0.40, 0.25, 0.15, 0.12, 0.08) for Group1..Group5
#                               (source stays equal): the perturbed shared population dominates the target, the
#                               population that is unmatched in the partial cases (Group5) is rare there
# Ablation controls, one factor each on top of L2:
#   A4_perturb_only             L2 + the L4 perturbation (no batch)
#   A5_imbalance_only           L2 + the L5 target proportions (no batch, no perturbation)
# D = 2 * de.prob * (de.facLoc^2 + de.facScale^2) is the expected squared log distance between two Splatter groups
# per gene, read from the installed Splatter's parameters.
#
# Pairing: the caller passes L2's seed for L3-L5 and the ablations, so they are the L2 simulation plus their factors:
# batch factors are drawn in every call whatever their scale, the perturbation factors come from their own seed
# (seed + 7919) before Splatter runs, and group sampling with 5 groups draws one uniform per cell whatever the
# probabilities; the deep / shallow calls must still agree on every gene-level truth column or the script stops.
# Writes the native files (counts.mtx, cells.csv, genes.csv, generation.json, GENERATION_DONE) plus
#   genes_truth.csv   gene_id, GeneMean, BaseGeneMean, OutlierFactor, DEFacGroup1..K, BatchFacBatch1, BatchFacBatch2,
#                     PerturbationFactor (1 = untouched)
#   cells_truth.csv   cell_id, side, batch, group, ExpLibSize, perturbed

suppressPackageStartupMessages({ library(splatter); library(Matrix); library(SingleCellExperiment) })
args <- commandArgs(trailingOnly = TRUE)
value_of <- function(flag, default = NULL) { hit <- which(args == flag); if (length(hit) == 0) return(default); args[hit[1] + 1] }
out <- value_of("--out"); if (is.null(out)) stop("--out is required")
n_cells <- as.integer(value_of("--n", "5000"))
replicate <- as.integer(value_of("--replicate", "1"))
level <- value_of("--level", "L0_matched")
groups <- as.integer(value_of("--groups", "5"))
seed <- as.integer(value_of("--seed", "20260925"))
param_set <- value_of("--param-set", "benchmark")
if (!(param_set %in% c("benchmark", "splatter_default"))) stop("--param-set must be benchmark or splatter_default")
pairing <- value_of("--pairing", "none")
if (!(pairing %in% c("none", "paired"))) stop("--pairing must be none or paired")
if (pairing == "paired" && param_set != "splatter_default") stop("--pairing paired needs --param-set splatter_default")
lib_loc <- as.numeric(value_of("--lib-loc", "11.0"))
lib_scale <- as.numeric(value_of("--lib-scale", "0.2"))
depth_ratio <- as.numeric(value_of("--depth-ratio", "0.50"))
dropout_mid_target <- as.numeric(value_of("--dropout-mid-target", "1.0"))
dropout_shape <- as.numeric(value_of("--dropout-shape", "-1"))
perturb_group <- "Group1"
IMBALANCE <- c(0.40, 0.25, 0.15, 0.12, 0.08)
SPEC <- list(
  L0_matched = list(depth = FALSE, dropout = FALSE, batch = 0, perturb = 0, imbalance = FALSE),
  L1_depth = list(depth = TRUE, dropout = FALSE, batch = 0, perturb = 0, imbalance = FALSE),
  L2_depth_dropout = list(depth = TRUE, dropout = TRUE, batch = 0, perturb = 0, imbalance = FALSE),
  L3_batch = list(depth = TRUE, dropout = TRUE, batch = 0.5, perturb = 0, imbalance = FALSE),
  L4_batch_perturb = list(depth = TRUE, dropout = TRUE, batch = 0.5, perturb = 0.5, imbalance = FALSE),
  L5_batch_perturb_imbalance = list(depth = TRUE, dropout = TRUE, batch = 0.5, perturb = 0.5, imbalance = TRUE),
  A4_perturb_only = list(depth = TRUE, dropout = TRUE, batch = 0, perturb = 0.5, imbalance = FALSE),
  A5_imbalance_only = list(depth = TRUE, dropout = TRUE, batch = 0, perturb = 0, imbalance = TRUE))
if (!(level %in% names(SPEC))) stop("--level must be one of ", paste(names(SPEC), collapse = ", "))
if (groups != length(IMBALANCE) && SPEC[[level]]$imbalance) stop("the imbalance is defined for 5 groups")
sp <- SPEC[[level]]
if (file.exists(file.path(out, "GENERATION_DONE"))) stop(paste(out, "already generated; not overwritten"))
dir.create(out, recursive = TRUE, showWarnings = FALSE)

target_ratio <- if (sp$depth) depth_ratio else 1.0
ref <- newSplatParams()
if (param_set == "splatter_default") {
  lib_loc <- getParam(ref, "lib.loc"); lib_scale <- getParam(ref, "lib.scale")
  dropout_mid_target <- getParam(ref, "dropout.mid"); dropout_shape <- getParam(ref, "dropout.shape")
}
lib_loc_target <- lib_loc + log(target_ratio)
target_dropout <- if (sp$dropout) dropout_mid_target else NA_real_
source_dropout <- if (sp$dropout && param_set == "splatter_default") dropout_mid_target else NA_real_
de_prob <- getParam(ref, "de.prob"); de_loc <- getParam(ref, "de.facLoc"); de_scale <- getParam(ref, "de.facScale")
de_down <- getParam(ref, "de.downProb")
D <- 2 * de_prob * (de_loc^2 + de_scale^2)
batch_fac_scale <- sqrt(sp$batch * D / 2); batch_fac_loc <- 0
if (param_set == "splatter_default") {
  batch_fac_loc <- if (sp$batch > 0 || pairing == "paired") getParam(ref, "batch.facLoc") else 0
  batch_fac_scale <- if (sp$batch > 0 || pairing == "paired") getParam(ref, "batch.facScale") else 0
}
batch_neutralised <- pairing == "paired" && sp$batch == 0
target_prob <- if (sp$imbalance) IMBALANCE else rep(1 / groups, groups)
cat(sprintf("splatter %s, %s, level %s: D %.5f, batch.facLoc %.4f, batch.facScale %.5f, dropout mid %s shape %s (source %s), perturbation gene probability %.4f, target group.prob %s\n",
            as.character(packageVersion("splatter")), param_set, level, D, batch_fac_loc, batch_fac_scale, target_dropout, dropout_shape,
            if (is.na(source_dropout)) "none" else "same", 2 * sp$perturb * de_prob,
            paste(round(target_prob, 3), collapse = "/")))

base_params <- function(cells_per_batch, prob) {
  setParams(newSplatParams(), batchCells = cells_per_batch, batch.facLoc = batch_fac_loc, batch.facScale = batch_fac_scale,
            group.prob = prob, lib.scale = lib_scale, seed = seed)
}
n_genes <- getParam(base_params(c(n_cells, n_cells), rep(1 / groups, groups)), "nGenes")
set.seed(seed + 7919L)
pert_factor <- rep(1, n_genes)
if (sp$perturb > 0) {
  hit <- runif(n_genes) < 2 * sp$perturb * de_prob
  sgn <- ifelse(runif(n_genes) < de_down, -1, 1)
  mag <- rnorm(n_genes, de_loc, de_scale)
  pert_factor[hit] <- exp(sgn[hit] * mag[hit])
}
st <- new.env(); st$active <- FALSE; st$assay <- NA_character_; st$cells <- 0L; st$factor <- pert_factor; st$group <- perturb_group
assign(".rb_perturb_state", st, envir = globalenv())
inject <- quote({
  s_ <- get(".rb_perturb_state", envir = globalenv())
  if (isTRUE(s_$active)) {
    nm_ <- intersect(c("BaseCellMeans", "CellMeans"), SummarizedExperiment::assayNames(sim))[1]
    if (is.na(nm_)) stop("no cell-mean assay to perturb")
    cols_ <- which(as.character(SummarizedExperiment::colData(sim)$Batch) == "Batch2" & as.character(SummarizedExperiment::colData(sim)$Group) == s_$group)
    m_ <- SummarizedExperiment::assay(sim, nm_); m_[, cols_] <- m_[, cols_] * s_$factor; SummarizedExperiment::assay(sim, nm_) <- m_
    s_$assay <- nm_; s_$cells <- length(cols_)
  }
})
suppressMessages(trace("splatSimBCVMeans", tracer = inject, where = asNamespace("splatter"), print = FALSE))
# drawn batch factors: recorded in every call; set to 1 before they are applied when the level has no batch (paired)
bst <- new.env(); bst$neutralise <- batch_neutralised; bst$drawn <- list()
assign(".rb_batch_state", bst, envir = globalenv())
batch_hook <- quote({
  b_ <- get(".rb_batch_state", envir = globalenv())
  rd_ <- SummarizedExperiment::rowData(sim)
  bc_ <- grep("^BatchFacBatch", colnames(rd_), value = TRUE)
  b_$drawn[[length(b_$drawn) + 1]] <- as.data.frame(rd_[, bc_, drop = FALSE])
  if (isTRUE(b_$neutralise)) {
    for (c_ in bc_) rd_[[c_]] <- rep(1, nrow(rd_))
    SummarizedExperiment::rowData(sim) <- rd_
  }
})
suppressMessages(trace("splatSimBatchCellMeans", tracer = batch_hook, where = asNamespace("splatter"), print = FALSE))
both_sides <- function(this_lib_loc, this_dropout, prob, perturb) {
  params <- setParams(base_params(c(n_cells, n_cells), prob), lib.loc = this_lib_loc)
  if (is.na(this_dropout)) params <- setParams(params, dropout.type = "none") else {
    params <- setParams(params, dropout.mid = this_dropout, dropout.shape = dropout_shape)
    params <- setParams(params, dropout.type = "experiment")
  }
  st$active <- perturb && sp$perturb > 0
  on.exit(st$active <- FALSE)
  splatSimulate(params, method = "groups", verbose = FALSE)
}
deep <- both_sides(lib_loc, source_dropout, rep(1 / groups, groups), FALSE)
shallow <- both_sides(lib_loc_target, target_dropout, target_prob, TRUE)
untrace("splatSimBCVMeans", where = asNamespace("splatter"))
untrace("splatSimBatchCellMeans", where = asNamespace("splatter"))
if (length(bst$drawn) != 2 || !identical(bst$drawn[[1]], bst$drawn[[2]])) stop("the two calls drew different batch factors")

crn_counts <- function(sim, crn_seed, this_dropout, keep_cols) {
  # paired sampler (see --pairing): Splatter's model on its BaseCellMeans, inverse CDF of common uniforms
  base <- SummarizedExperiment::assay(sim, "BaseCellMeans")
  G <- nrow(base); C <- ncol(base)
  bcv_common <- getParam(ref, "bcv.common"); bcv_df <- getParam(ref, "bcv.df")
  set.seed(crn_seed)
  chisq <- qchisq(runif(G), df = bcv_df)
  bcv <- (bcv_common + 1 / sqrt(base)) * sqrt(bcv_df / chisq)
  means <- matrix(qgamma(runif(G * C), shape = 1 / (bcv^2), scale = base * (bcv^2)), nrow = G)
  rm(bcv); means[base == 0 | !is.finite(means)] <- 0
  counts <- matrix(qpois(runif(G * C), lambda = means), nrow = G)
  u_drop <- runif(G * C)
  if (!is.na(this_dropout)) {
    p_drop <- 1 / (1 + exp(-dropout_shape * (log(means) - this_dropout)))
    counts <- counts * (u_drop < (1 - p_drop))
    rm(p_drop)
  }
  rm(u_drop)
  spl <- as.matrix(BiocGenerics::counts(sim))[, keep_cols, drop = FALSE]; ours <- counts[, keep_cols, drop = FALSE]
  check <- data.frame(gene_id = rownames(sim), mean_splatter = rowMeans(spl), mean_paired = rowMeans(ours),
                      zero_splatter = rowMeans(spl == 0), zero_paired = rowMeans(ours == 0))
  dimnames(counts) <- dimnames(base)
  BiocGenerics::counts(sim) <- counts
  list(sim = sim, check = check)
}
sampler_check <- NULL
if (pairing == "paired") {
  d_ <- crn_counts(deep, seed + 271829L, source_dropout, as.character(colData(deep)$Batch) == "Batch1")
  s_ <- crn_counts(shallow, seed + 271830L, target_dropout, as.character(colData(shallow)$Batch) == "Batch2")
  deep <- d_$sim; shallow <- s_$sim
  sampler_check <- rbind(cbind(side = "source", d_$check), cbind(side = "target", s_$check))
  write.csv(sampler_check, file.path(out, "sampler_check.csv"), row.names = FALSE)
  rm(d_, s_); gc()
}
if (sp$perturb > 0 && st$cells == 0) stop("perturbation requested but no target cell of ", perturb_group)

source_sim <- deep[, as.character(colData(deep)$Batch) == "Batch1"]
target_sim <- shallow[, as.character(colData(shallow)$Batch) == "Batch2"]
rd_deep <- as.data.frame(rowData(deep)); rd_shallow <- as.data.frame(rowData(shallow))
truth_cols <- grep("^(GeneMean|BaseGeneMean|OutlierFactor|DEFacGroup|BatchFacBatch)", colnames(rd_deep), value = TRUE)
for (nm in truth_cols) if (!identical(rd_deep[[nm]], rd_shallow[[nm]])) stop(sprintf("deep and shallow calls differ in %s: not the same biology", nm))
src_groups <- as.character(colData(source_sim)$Group); tgt_groups <- as.character(colData(target_sim)$Group)
if (!identical(sort(unique(src_groups)), sort(unique(tgt_groups)))) stop("the sides do not carry the same populations")

counts_all <- cbind(counts(source_sim), counts(target_sim))
cell_ids <- c(paste0("Batch1_", colnames(source_sim)), paste0("Batch2_", colnames(target_sim)))
colnames(counts_all) <- cell_ids
batch_labels <- c(rep("Batch1", ncol(source_sim)), rep("Batch2", ncol(target_sim)))
group_labels <- c(src_groups, tgt_groups)
writeMM(as(counts_all, "CsparseMatrix"), file.path(out, "counts.mtx"))
write.csv(data.frame(cell_id = cell_ids, batch = batch_labels, group = group_labels), file.path(out, "cells.csv"), row.names = FALSE)
write.csv(data.frame(gene_id = rownames(source_sim), gene_mean = as.numeric(rd_deep$GeneMean)), file.path(out, "genes.csv"), row.names = FALSE)
drawn <- bst$drawn[[1]]; colnames(drawn) <- sub("^BatchFac", "BatchFacDrawn", colnames(drawn))
write.csv(data.frame(gene_id = rownames(source_sim), rd_deep[, truth_cols, drop = FALSE], drawn, PerturbationFactor = pert_factor),
          file.path(out, "genes_truth.csv"), row.names = FALSE)
write.csv(data.frame(cell_id = cell_ids, side = ifelse(batch_labels == "Batch1", "source", "target"), batch = batch_labels,
                     group = group_labels,
                     ExpLibSize = c(as.numeric(colData(source_sim)$ExpLibSize), as.numeric(colData(target_sim)$ExpLibSize)),
                     perturbed = batch_labels == "Batch2" & group_labels == perturb_group & sp$perturb > 0),
          file.path(out, "cells_truth.csv"), row.names = FALSE)
depth <- Matrix::colSums(counts_all); detected <- Matrix::colSums(counts_all > 0); is_source <- batch_labels == "Batch1"
record <- list(
  generator = "realdata_benchmark/splatter_n5000/20_generate_n5000.R", route = "D", n_cells_per_side = n_cells,
  n_semantics = "cells per side (batchCells = c(N, N)), as the native benchmark", groups = groups, replicate = replicate,
  technical_level = level, param_set = param_set, seed = seed, depth = sp$depth, dropout = sp$dropout, lib_loc_source = lib_loc,
  lib_loc_target = lib_loc_target, lib_scale = lib_scale, dropout_mid_target = target_dropout, dropout_mid_source = source_dropout,
  dropout_shape = dropout_shape, batch_strength = sp$batch, batch_fac_loc = batch_fac_loc, batch_fac_scale = batch_fac_scale,
  perturb_strength = sp$perturb, perturb_group = perturb_group, perturb_gene_probability = 2 * sp$perturb * de_prob,
  perturbed_genes = sum(pert_factor != 1), perturbed_assay = st$assay, perturbed_target_cells = st$cells, perturb_seed = seed + 7919L,
  target_group_prob = target_prob, source_group_prob = rep(1 / groups, groups),
  group_de_distance_D = D, de_prob = de_prob, de_fac_loc = de_loc, de_fac_scale = de_scale, de_down_prob = de_down,
  n_genes = n_genes, splatter_version = as.character(packageVersion("splatter")), r_version = R.version.string,
  pairing = pairing, batch_neutralised = batch_neutralised,
  count_sampler = if (pairing == "paired") "paired inverse-CDF sampler of Splatter's BCV / Poisson / dropout model (custom)" else "splatter",
  crn_seeds = if (pairing == "paired") c(seed + 271829L, seed + 271830L) else NULL,
  sampler_check_median_abs_log2_mean_ratio = if (is.null(sampler_check)) NULL else {
    k_ <- sampler_check$mean_splatter > 0.5; median(abs(log2(sampler_check$mean_paired[k_] / sampler_check$mean_splatter[k_]))) },
  sampler_check_max_abs_zero_fraction_difference = if (is.null(sampler_check)) NULL else max(abs(sampler_check$zero_paired - sampler_check$zero_splatter)),
  source_median_nCount = median(depth[is_source]), target_median_nCount = median(depth[!is_source]),
  source_median_nFeature = median(detected[is_source]), target_median_nFeature = median(detected[!is_source]),
  source_min_nFeature = min(detected[is_source]), target_min_nFeature = min(detected[!is_source]),
  target_group_counts = as.list(table(tgt_groups)))
writeLines(jsonlite::toJSON(record, auto_unbox = TRUE, pretty = TRUE, na = "null"), file.path(out, "generation.json"))
writeLines("SUCCESS", file.path(out, "GENERATION_DONE"))
cat(sprintf("wrote %s: %d cells per side, %s; %d perturbed genes on %d target cells (%s)\n",
            out, n_cells, level, sum(pert_factor != 1), st$cells, st$assay))
