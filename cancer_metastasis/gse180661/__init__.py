"""GSE180661 patient-pair ConfidenceOT analysis workflow.

The universal unit is one patient, one primary sample/site, and one metastatic
sample/site. Modules cover cohort inventory, all-cell transition validation,
malignant-cell mapping, primary-only pseudobulk/DEG/GSEA, and frozen-signature
TCGA-OV survival analysis. TACC entry points live in ``slurm``.
"""
