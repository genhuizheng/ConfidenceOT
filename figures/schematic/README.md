# Schematics

Every figure whose job is to explain rather than to report. Results stay in
`benchmark_results/`; nothing in this folder carries a measurement from a run.

The images are not tracked — they are rebuilt from the scripts, which are.
Each script now defaults to this folder and to the settings the figure was
agreed at, so the command that makes a figure is the bare command. Five
directories of near-identical scenario renders existed because that
invocation used to live in a shell history instead of in the file.

| file | what it says | rebuild with |
|---|---|---|
| `fig_confidenceot_schematic` | Figure 1: the method. Two samples that need not describe the same populations, a transport plan, and a keep/reject label on every point. | `python scripts/make_method_schematic.py` |
| `scenario_umap` | The biological question: source and target side by side on one embedding, a population that is gone in S1 and one that is new in S2. | `python scripts/make_scenario_umap.py` |
| `problem` | The two technical nuisances, each as the embedding one wants beside the embedding one gets. No preprocessing appears; the structure on the right is the artefact. | `python scripts/make_nuisance_umap.py` |
| `benchmark` | The benchmark setting: counts → OT coupling → rejection gate → ground-truth mask, for the depth arm and the gene-detection arm. Coupling and gate are schematic. | same script, same run |

`diagnostics.json` holds the numbers the panels were measured at — the
embedding-versus-nuisance correlations for `problem`, and the realised nCount
and nFeature ranges for `benchmark`. It is written by
`make_nuisance_umap.py`, and `make_scenario_umap.py` overwrites it with its
own, so read it straight after the run that produced it.

The scoring is not here: metrics are defined in `BENCHMARK_METRICS.md`, so
that a number cannot be chosen after seeing a plot.

Nothing else in the repository is a schematic. `figure2/panel_e_schematic`
looked like one, but `make_figure2.py` stopped producing it when
`panel_e_validation` took slot (e); the files were a leftover of the removed
panel and have been deleted rather than moved.
