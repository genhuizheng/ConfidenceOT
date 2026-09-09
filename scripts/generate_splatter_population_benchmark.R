#!/usr/bin/env Rscript

# Generate S0--S5 paired-population benchmarks from fresh Splatter pools.
# Each output contains exactly N source and N target cells. Scenarios derived
# from the same (N, replicate) pool share the expression model, making the
# scenario comparison paired without reusing cells across repetitions.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L || length(args) > 7L) {
  stop(
    paste(
      "Usage: generate_splatter_population_benchmark.R OUTPUT",
      "[REPEATS N_GENES BASE_SEED SIZES_CSV BATCH_FAC_LOC BATCH_FAC_SCALE]"
    ),
    call. = FALSE
  )
}

output_root <- normalizePath(args[[1L]], winslash = "/", mustWork = FALSE)
repeats <- if (length(args) >= 2L) as.integer(args[[2L]]) else 3L
n_genes <- if (length(args) >= 3L) as.integer(args[[3L]]) else 1000L
base_seed <- if (length(args) >= 4L) as.integer(args[[4L]]) else 7300L
sizes <- if (length(args) >= 5L) as.integer(strsplit(args[[5L]], ",", fixed = TRUE)[[1L]]) else c(100L, 500L, 1000L)
batch_fac_loc <- if (length(args) >= 6L) as.numeric(args[[6L]]) else 0
batch_fac_scale <- if (length(args) >= 7L) as.numeric(args[[7L]]) else 0
batch_condition <- if (batch_fac_loc == 0 && batch_fac_scale == 0) "none" else "mild"

project_library <- normalizePath(file.path(getwd(), ".r-library"), winslash = "/", mustWork = FALSE)
if (dir.exists(project_library)) .libPaths(c(project_library, .libPaths()))
needed <- c("splatter", "SingleCellExperiment", "SummarizedExperiment", "Matrix")
missing <- needed[!vapply(needed, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing)) stop("Missing R packages: ", paste(missing, collapse = ", "), call. = FALSE)

if (!is.finite(repeats) || repeats < 1L || !is.finite(n_genes) || n_genes < 200L ||
    any(!is.finite(sizes)) || any(sizes < 30L) || !is.finite(batch_fac_loc) ||
    !is.finite(batch_fac_scale) || batch_fac_scale < 0) {
  stop("Invalid repeats, gene count, or cell sizes.", call. = FALSE)
}

scenarios <- c("S0_clean_movement", "S1_extinction", "S2_emergence",
               "S3_source_outlier", "S4_bifurcation", "S5_abundance_shift")
base_populations <- LETTERS[1:7]

integer_quotas <- function(n, weights, names) {
  raw <- n * weights / sum(weights)
  q <- floor(raw)
  remainder <- n - sum(q)
  if (remainder > 0L) {
    order_fraction <- order(raw - q, decreasing = TRUE, method = "radix")
    q[order_fraction[seq_len(remainder)]] <- q[order_fraction[seq_len(remainder)]] + 1L
  }
  stats::setNames(as.integer(q), names)
}

select_by_quota <- function(indices, groups, quotas) {
  selected <- integer(0L)
  labels <- character(0L)
  for (population in names(quotas)) {
    candidates <- indices[groups[indices] == population]
    needed_count <- quotas[[population]]
    if (length(candidates) < needed_count) {
      stop("Splatter pool has only ", length(candidates), " cells for ", population,
           "; need ", needed_count, ".", call. = FALSE)
    }
    chosen <- sample(candidates, needed_count, replace = FALSE)
    selected <- c(selected, chosen)
    labels <- c(labels, rep(population, needed_count))
  }
  permutation <- sample(seq_along(selected), length(selected), replace = FALSE)
  list(index = selected[permutation], population = labels[permutation])
}

add_module <- function(counts, columns, genes, rate = 0.15) {
  if (!length(columns) || !length(genes)) return(counts)
  means <- Matrix::rowMeans(counts[genes, columns, drop = FALSE])
  extra <- matrix(
    stats::rpois(length(genes) * length(columns), lambda = rep(rate * pmax(means, 0.1), length(columns))),
    nrow = length(genes)
  )
  counts[genes, columns] <- counts[genes, columns, drop = FALSE] + extra
  counts
}

write_scenario <- function(counts_pool, gene_ids, source, target, scenario, n, replicate_id, seed) {
  selected <- c(source$index, target$index)
  counts <- counts_pool[, selected, drop = FALSE]
  population <- c(source$population, target$population)
  condition <- c(rep("source", n), rep("target", n))

  # Controlled shifts that preserve or split population identity.
  expressed <- order(Matrix::rowMeans(counts), decreasing = TRUE)
  module_size <- min(30L, length(expressed))
  if (scenario == "S0_clean_movement") {
    b_target <- which(condition == "target" & population == "B")
    counts <- add_module(counts, b_target, expressed[seq_len(module_size)], rate = 0.18)
  }
  if (scenario == "S4_bifurcation") {
    b_target <- which(condition == "target" & population == "B")
    split <- sample(rep(c("B1", "B2"), length.out = length(b_target)))
    population[b_target] <- split
    first_genes <- expressed[seq_len(module_size)]
    second_genes <- expressed[module_size + seq_len(module_size)]
    counts <- add_module(counts, b_target[split == "B1"], first_genes, rate = 0.20)
    counts <- add_module(counts, b_target[split == "B2"], second_genes, rate = 0.20)
  }
  if (scenario == "S3_source_outlier") {
    population[condition == "source" & population == "G"] <- "O"
  }

  expected_rejection <- rep(FALSE, 2L * n)
  if (scenario == "S1_extinction") expected_rejection[condition == "source" & population == "A"] <- TRUE
  if (scenario == "S2_emergence") expected_rejection[condition == "target" & population == "G"] <- TRUE
  if (scenario == "S3_source_outlier") expected_rejection[condition == "source" & population == "O"] <- TRUE

  cell_id <- sprintf("%s_n%04d_r%02d_cell_%05d", substr(scenario, 1L, 2L), n, replicate_id, seq_len(2L * n))
  colnames(counts) <- cell_id
  metadata <- data.frame(
    cell_id = cell_id,
    condition = condition,
    population = population,
    expected_rejection = expected_rejection,
    scenario = scenario,
    replicate = replicate_id,
    seed = seed,
    requested_cells_per_condition = n,
    stringsAsFactors = FALSE
  )
  destination <- file.path(output_root, scenario, sprintf("n_%04d", n), sprintf("rep_%02d", replicate_id))
  dir.create(destination, recursive = TRUE, showWarnings = FALSE)
  Matrix::writeMM(Matrix::Matrix(counts, sparse = TRUE), file.path(destination, "counts.mtx"))
  utils::write.csv(metadata, file.path(destination, "cells.csv"), row.names = FALSE)
  utils::write.table(data.frame(gene_id = gene_ids), file.path(destination, "genes.tsv"),
                     sep = "\t", row.names = FALSE, quote = FALSE)
  manifest <- data.frame(
    key = c("generator", "splatter_version", "scenario", "replicate", "seed",
            "cells_per_condition", "genes", "pool_cells_per_condition", "population_truth",
            "batch_condition", "batch_fac_loc", "batch_fac_scale", "batch_effect_model"),
    value = c("fresh_splatSimulateGroups_paired_scenario_derivation",
              as.character(utils::packageVersion("splatter")), scenario, replicate_id, seed,
              n, nrow(counts), 3L * n,
              "S1 source A; S2 target G; S3 source O; S0/S4/S5 none",
              batch_condition, batch_fac_loc, batch_fac_scale,
              "Splatter native log-normal batch factors applied to all genes"),
    stringsAsFactors = FALSE
  )
  utils::write.csv(manifest, file.path(destination, "manifest.csv"), row.names = FALSE)
}

dir.create(output_root, recursive = TRUE, showWarnings = FALSE)
for (n in sizes) {
  for (replicate_id in seq_len(repeats)) {
    # Preserve the historical N=100/500/1000 seed schedule when larger sizes
    # are requested, so regenerated N=1000 cases remain directly comparable.
    seed_sizes <- sort(unique(c(100L, 500L, 1000L, sizes)))
    seed <- base_seed + 1000L * match(n, seed_sizes) + replicate_id
    set.seed(seed)
    pool_n <- 3L * n
    message("Splatter pool: N=", n, ", replicate=", replicate_id, ", seed=", seed)
    sim <- splatter::splatSimulateGroups(
      batchCells = c(pool_n, pool_n),
      nGenes = n_genes,
      group.prob = rep(1 / 7, 7),
      de.prob = rep(0.12, 7),
      de.facLoc = rep(0.10, 7),
      de.facScale = rep(0.35, 7),
      batch.facLoc = batch_fac_loc,
      batch.facScale = batch_fac_scale,
      seed = seed,
      verbose = FALSE
    )
    counts_pool <- SingleCellExperiment::counts(sim)
    meta <- as.data.frame(SummarizedExperiment::colData(sim))
    batch <- as.character(meta$Batch)
    group_number <- as.integer(sub("Group", "", as.character(meta$Group), fixed = TRUE))
    groups <- base_populations[group_number]
    batch_levels <- unique(batch)
    source_pool <- which(batch == batch_levels[[1L]])
    target_pool <- which(batch == batch_levels[[2L]])
    gene_ids <- rownames(counts_pool)
    if (is.null(gene_ids)) gene_ids <- sprintf("gene_%05d", seq_len(nrow(counts_pool)))

    equal6 <- integer_quotas(n, rep(1, 6), LETTERS[1:6])
    equal5 <- integer_quotas(n, rep(1, 5), LETTERS[2:6])
    equal7 <- integer_quotas(n, rep(1, 7), LETTERS[1:7])
    source_outlier <- integer_quotas(n, c(rep(0.15, 6), 0.10), LETTERS[1:7])
    abundance_source <- equal6
    abundance_target <- integer_quotas(n, c(0.30, 0.05, rep(0.1625, 4)), LETTERS[1:6])

    selections <- list(
      S0_clean_movement = list(equal6, equal6),
      S1_extinction = list(equal6, equal5),
      S2_emergence = list(equal6, equal7),
      S3_source_outlier = list(source_outlier, equal6),
      S4_bifurcation = list(equal6, equal6),
      S5_abundance_shift = list(abundance_source, abundance_target)
    )
    for (scenario in scenarios) {
      quota_pair <- selections[[scenario]]
      source <- select_by_quota(source_pool, groups, quota_pair[[1L]])
      target <- select_by_quota(target_pool, groups, quota_pair[[2L]])
      write_scenario(counts_pool, gene_ids, source, target, scenario, n, replicate_id, seed)
    }
    rm(sim, counts_pool); gc(verbose = FALSE)
  }
}

manifest <- data.frame(
  key = c("generator", "splatter_version", "repeats", "sizes", "n_genes", "base_seed",
          "scenarios", "cells_are_per_condition", "batch_condition", "batch_fac_loc",
          "batch_fac_scale", "batch_effect_model"),
  value = c("fresh_splatSimulateGroups_paired_scenario_derivation",
            as.character(utils::packageVersion("splatter")), repeats,
            paste(sizes, collapse = ","), n_genes, base_seed,
            paste(scenarios, collapse = ","), "TRUE", batch_condition, batch_fac_loc,
            batch_fac_scale, "Splatter native log-normal batch factors applied to all genes"),
  stringsAsFactors = FALSE
)
utils::write.csv(manifest, file.path(output_root, "manifest.csv"), row.names = FALSE)
message("Wrote Splatter population benchmark to ", output_root)
