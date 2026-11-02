#!/usr/bin/env Rscript
# Splatter N = 5,000 benchmark, cumulative levels L0-L5, built on the native generator
# (benchmark/tacc/10_generate_splatter_native.R, route D: same seeds, same route, same output files).
#
#   benchmark/tacc/rscript.sh realdata_benchmark/splatter_n5000/20_generate_n5000.R --out DIR --n 5000 \
#       --replicate R --level LEVEL --seed S
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
target_dropout <- if (sp$dropout) dropout_mid_target else NA_real_
lib_loc_target <- lib_loc + log(target_ratio)
ref <- newSplatParams()
de_prob <- getParam(ref, "de.prob"); de_loc <- getParam(ref, "de.facLoc"); de_scale <- getParam(ref, "de.facScale")
de_down <- getParam(ref, "de.downProb")
D <- 2 * de_prob * (de_loc^2 + de_scale^2)
batch_fac_scale <- sqrt(sp$batch * D / 2)
target_prob <- if (sp$imbalance) IMBALANCE else rep(1 / groups, groups)
cat(sprintf("splatter %s, level %s: D %.5f, batch.facScale %.5f, perturbation gene probability %.4f, target group.prob %s\n",
            as.character(packageVersion("splatter")), level, D, batch_fac_scale, 2 * sp$perturb * de_prob,
            paste(round(target_prob, 3), collapse = "/")))

base_params <- function(cells_per_batch, prob) {
  setParams(newSplatParams(), batchCells = cells_per_batch, batch.facLoc = 0, batch.facScale = batch_fac_scale,
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
deep <- both_sides(lib_loc, NA_real_, rep(1 / groups, groups), FALSE)
shallow <- both_sides(lib_loc_target, target_dropout, target_prob, TRUE)
untrace("splatSimBCVMeans", where = asNamespace("splatter"))
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
write.csv(data.frame(gene_id = rownames(source_sim), rd_deep[, truth_cols, drop = FALSE], PerturbationFactor = pert_factor),
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
  technical_level = level, seed = seed, depth = sp$depth, dropout = sp$dropout, lib_loc_source = lib_loc,
  lib_loc_target = lib_loc_target, lib_scale = lib_scale, dropout_mid_target = target_dropout, dropout_shape = dropout_shape,
  batch_strength = sp$batch, batch_fac_loc = 0, batch_fac_scale = batch_fac_scale,
  perturb_strength = sp$perturb, perturb_group = perturb_group, perturb_gene_probability = 2 * sp$perturb * de_prob,
  perturbed_genes = sum(pert_factor != 1), perturbed_assay = st$assay, perturbed_target_cells = st$cells, perturb_seed = seed + 7919L,
  target_group_prob = target_prob, source_group_prob = rep(1 / groups, groups),
  group_de_distance_D = D, de_prob = de_prob, de_fac_loc = de_loc, de_fac_scale = de_scale, de_down_prob = de_down,
  n_genes = n_genes, splatter_version = as.character(packageVersion("splatter")), r_version = R.version.string,
  source_median_nCount = median(depth[is_source]), target_median_nCount = median(depth[!is_source]),
  source_median_nFeature = median(detected[is_source]), target_median_nFeature = median(detected[!is_source]),
  source_min_nFeature = min(detected[is_source]), target_min_nFeature = min(detected[!is_source]),
  target_group_counts = as.list(table(tgt_groups)))
writeLines(jsonlite::toJSON(record, auto_unbox = TRUE, pretty = TRUE, na = "null"), file.path(out, "generation.json"))
writeLines("SUCCESS", file.path(out, "GENERATION_DONE"))
cat(sprintf("wrote %s: %d cells per side, %s; %d perturbed genes on %d target cells (%s)\n",
            out, n_cells, level, sum(pert_factor != 1), st$cells, st$assay))
