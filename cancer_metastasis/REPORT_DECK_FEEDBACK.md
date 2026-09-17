# Report deck review log

Feedback on `ConfidenceOT_report_20260916.pptx` and what was done about it.
Page numbers are the ones printed in the orange block at the bottom right, which
is also the order the generator emits.

The deck is generated, not hand-edited, so every change here is a change to
`cancer_metastasis/tools/build_report_deck.py` and survives a rebuild.

---

## Round 1 — slide 2 (the overview), 2026-09-16

| # | Comment | Status | What changed |
|---|---|---|---|
| 1.1 | 要说明一下生物学问题究竟是什么 | done | The overview split into two pages. Page 2 is now the biological question with no method vocabulary in it: which cells in a primary tumour can leave and seed a distant site, what follows if we can name them, and why paired primary–metastasis data is what makes the question answerable. |
| 1.2 | 什么方向是反的?Effect 的定义是什么?P value 的什么的?systematic artefact?啥是 artefact | done | A glossary block sits beside the conclusions on page 3, defining effect, p value, artefact and confounder in plain language, plus why a tiny effect with an overwhelming p value is the signature of an artefact rather than of small biology. |
| 1.3 | 这句话就是我们的 assumption。如果用 OT 找出了相似的细胞,或许他们就是具有转移潜力的细胞。1 回答的问题,2 就是 logic,也就是 4 改成 2:研究方法 | done | Reordered as asked. The assumption is now stated as an assumption on page 2, item 1 of the second block: metastatic cells are descendants of primary cells, so primary cells that look transcriptionally like the metastasis may be the ones with metastatic potential. The method follows it as item 2. The similarity finding moved into the results, where it reads as "the premise is not supported" rather than as a stray negative. |
| 1.4 | 后面我们会用 cap 来操作,就是最多拒绝 85% 细胞 | done | Page 3's prostate item now says the gene-level analysis uses the capped rule — at most 85% of cells rejected, so at least 15% retained — because 1.6% leaves too few cells for statistics in most patients. |

## Round 2 — slides 4 and 5, 2026-09-16

| # | Comment | Status | What changed |
|---|---|---|---|
| 2.1 | 为啥图完全不一样?这个要 check,可能是重大问题 (two UMAPs side by side) | done | The two images are different datasets — ovarian GSE180661 against colorectal GSE225857 — so the embeddings differ by construction. But the instinct was right about something real: in the pooled ovarian figure the kept cells come in whole blocks, and the split is substantially patient identity. **45.6% of the variance in the retained label is between patients**, with 8 of 29 patients below 5% kept and 5 above 80%. In prostate it is 1.2%, because the cap forces every patient to about 15%. So the visible difference between the two figures is largely the cap: with one, every patient is pushed to the same fraction and the split must happen inside patients; without one, patients run to 0 or 0.93. |
| 2.2 | 最好挑一个病人说明,就一个病人,然后放附图就是整体的 | done | SPECTRUM-OV-083 is the main figure: 4,726 primary and 4,307 metastatic cells, 54% kept, so both groups are visible. Chosen for legibility, not for its result; its cell-division gap of +0.126 has the same sign as the cohort median of +0.089. Inside that one patient the split follows position in the embedding rather than taking the whole cluster, which is what the pooled figure could not show. The pooled view is now a single-panel inset labelled 附图, with the 45.6% beside it. |
| 2.3 | 下面那两段 paragraph 没有意义,最好做个示意图:A B population 是 primary,A 的技术差异 nFeature 高、B 低,C 是 metastasis,由于技术性差异 A 更靠近,说明技术性差异 > 生物学差异 | done | Built as `fig_depth_cartoon.png` and given its own page, replacing the prose. Two panels: the same biology, then the same two populations after depth is added, with A pulled toward C. The depths on it, 13,800 against 10,400, are the real group medians. |
| 2.4 | depth 究竟是什么?要说明清楚 | done | A third column on that schematic defines it: the reads a cell yields (total counts) and the genes detected from them (nFeature), why it varies between cells (capture efficiency, library prep, sequencing batch — the pipeline, not the cell), and why it moves distance (a shallow cell reads zero for genes it does express, so its profile is missing a chunk and looks unlike C even when its biology is identical). |
| 2.5 | 在这一张前面插入一个图,rank 是什么方法,pearson 是什么方法,downsample 是什么方法 | done | `fig_methods_cartoon.png`, its own page before the depth results. Two cells with identical biology over six genes, one sequenced 1.3× deeper, and what each of log-CPM, downsample, gene rank and Pearson residuals leaves behind. The sixth gene carries it: lowly expressed, detected in the deep cell and zero in the shallow one, which is the part dividing by the cell total cannot fix. Marked 有效 / 无效, with the caption stating that the two that work have to be used together — rank alone on raw depth gives AUC 0.06. |

## Round 3 — the batch-effect question, 2026-09-16

| # | Comment | Status | What changed |
|---|---|---|---|
| 3.1 | 这些 retain / reject 是来自于同一个数据集吗?因为 primary 可能有多个,可能是批次效应 | done, and it checks out | All cells are from one study, GSE180661. The concern is still the right one to raise, because 16 of 29 patients have primary tissue from two anatomical sites, and two sites means two specimens and two libraries — so site and batch are confounded, and a test on site bounds both at once. It was tested and the split does not follow site. New page 5 carries the answer. |

The three-way decomposition of the retained label, over 84,465 primary cells:

| Source | Share of variance |
|---|---:|
| Between patients | 45.6% |
| **Between primary sites within a patient** | **5.0%** |
| Cell to cell within one sample | 49.4% |

Per patient, site explains a median of 4.5% of the within-patient split and
never more than 50% in any of the 16. The largest is SPECTRUM-OV-050 at 30%
(its two sites split 0.21 against 0.85, and that one is worth remembering).
SPECTRUM-OV-083, the patient on page 4, is 5.6%, so that figure is not a site
effect.

`sample_id` in the overlay holds the anatomical site — `left_adnexa`,
`right_ovary` and so on — not a sequencing batch id, which is why the test is
phrased as a bound on site and batch together rather than on batch alone.

## Round 4 — compression and two verification questions, 2026-09-16

| # | Comment | Status | What changed |
|---|---|---|---|
| 4.1 | slide 5 和 6 合并,slide 5 部分除了表格 text 都抛弃 | done | One page now, tables only. The three-level variance table sits beside the old rejection-rate table under a 两个检查 heading; slide 5's three prose items and the pooled-embedding figure are gone. |
| 4.2 | Gene rank 说清楚,pearson residual 也说清楚 | done | Both spelled out under the schematic as what is actually computed. Gene rank in three steps: divide each cell by its own total; divide each gene by that gene's nonzero median over both sides stacked; rank within the cell and keep the top 256, discarding the values. Pearson residuals as expected = cell total × the gene's share of the whole, residual = (observed − expected) / the standard deviation of the expected, with the note that depth enters the expectation and should cancel, but a gene observed at zero still leaves a large negative residual. |
| 4.3 | slide 11 还行 | kept | The simulation benchmark stays on its own page. It is the only place that can say a method is correct rather than that a number improved, because the simulation has a known answer. |
| 4.4 | slide 12 删除也行 | deleted, but not lost | The standalone mismatched-control page is gone. Its numbers moved onto the similarity page, which already plots the mismatched pairs as its second series: 0.351 → 0.000, p = 8e-17, gate Jaccard 0 at chance. That page was the only evidence the gate uses the metastasis at all, so the argument had to survive even though the page did not. |
| 4.5 | 15、16 是什么要说清楚,是 prostate cancer,17 是卵巢 | done | Every one of those pages now names its cancer in the heading. Note the ordering was one off: the cell-division gene page draws on the prostate DEG, and the ovarian figure is the page after it. Headings now read 前列腺:富集分析, 前列腺:原因二, and 卵巢与前列腺都一致（图为卵巢）. |
| 4.6 | 你是做了每个 patient primary vs metastasis 对吧,没有混在一起吧?没有 cap 约束? | answered, and it found something | Per patient, yes: one transport run per primary–metastasis pair, 94 pairs in ovarian and 24 in prostate, both sides always the same patient. Cross-patient pairing exists only as the deliberate control. On the cap, see below — the answer is not uniform across arms and one arm is worse than the deck was saying. |

### The cap is not inert in the prostate arm the gene-level results use

`forced_in_share_of_retained`, the share of retained cells that the coverage
floor put back rather than the rule selecting:

| Arm | retained | forced in by the cap | the rule alone would retain |
|---|---:|---:|---:|
| Ovarian rank256 + ds (free) | 0.292 | **0.0000** | 0.292 |
| Ovarian log-CPM (free) | 0.352 | **0.0000** | 0.352 |
| Prostate rank256 + ds (free) | 0.016 | **0.0000** | 0.016 |
| **Prostate cap 0.85** | 0.150 | **0.8411** | **0.024** |

Every ovarian page is from a free arm with nothing forced in, and ten of the
twenty-nine patients sit below 0.15, which a 0.85 rejection cap could not have
allowed. But the prostate capped arm — the one behind the GSEA and the
cell-division gene list — has **84% of its retained set put there by the cap**.
The rule on its own retains 2.4%, and `sign_rule_concordance` falls from 1.000
elsewhere to 0.874. The proliferation signal is still strong, but the group it
describes is mostly cap fill, and both of those pages now say so.

## Round 5 — one panel instead of two, 2026-09-16

| # | Comment | Status | What changed |
|---|---|---|---|
| 5.1 | 这个图更加明显一些,取代那张,这个是一眼看出来啥情况 (the old GSE225857 three-series panel) | done | The virtue of that figure is that all three groups are in **one** panel: the metastasis as a grey backdrop with both primary groups drawn over it, so where the retained cells sit *relative to the metastasis* can be read directly. The old three-panel version put the two primary groups in one panel and the metastasis in another, so that comparison was not on the page at all. Rebuilt as `fig_gatemap_*`: one panel with metastasis grey, retained orange, rejected aqua and counts in the legend, plus the cell-division panel kept beside it because it carries the proliferation argument. Retained keeps the orange it has throughout the deck; rejected takes the next slot in the validated order rather than the grey now doing duty as the backdrop. Applied to the ovarian patient page and the prostate page. |
| 5.2 | depth 那页要说明这个不是批次效应 | done | Added as its own block. Batch effects live between samples and libraries; depth is per cell and varies inside a single library, which is why batch correction of the Harmony kind cannot reach it — those methods align offsets between samples and this offset is within one. That is also why it needed downsampling specifically. The page points back to the 5.0% figure so the two questions read as separately settled: batch is not the problem, depth is. |

What this figure now shows on its own, for the featured patient: orange
concentrates in the lower-left tail, aqua sits in the upper body, and the
cell-division panel beside it puts its highest scores on exactly that tail.

## Round 6 — the colorectal figure, 2026-09-16

| # | Comment | Status | What changed |
|---|---|---|---|
| 6.1 | 也加上这个图,这个结肠癌的,有数据吗 | done, the data was local | `confidenceot_analysis/replication_umap/GSE225857/umap_coordinates_and_labels.csv.gz` holds every cell's coordinates, side and gate label, so the figure is redrawn in the deck's own colours rather than pasted in its original palette. New page 6. Head and neck is in the same directory if it is ever wanted, but 810 cells with 90 retained is thin. |

That export also carries the authors' eleven tumour subtypes, two of which are
proliferative, so it can test the cell-division finding against an independent
annotation in a third cancer. The second panel does that.

| Subtype group | n | retained |
|---|---:|---:|
| Tu05_PCNA and Tu07_MKI67 | 1,598 | **36.1%** |
| The other nine | 7,987 | 25.1% |

Odds ratio 1.68, p = 1.4e-18, and 3 of the 5 patients point the same way. Two
reasons it is on the slide as weak corroboration rather than as a result:

- The proliferative subtypes are not the top-ranked ones. Tu02_DEFA5 enriches
  2.8-fold and Tu11_PLA2G2A 2.2-fold, both above them, and Tu03_SRRM2 is
  depleted 2.6-fold. Proliferation ranks fifth and sixth.
- This is the old gate, cap enforced and depth uncorrected, and MKI67-positive
  cells carry more RNA and sequence deeper. On that arm depth and proliferation
  cannot be separated from each other, which is exactly what the depth-corrected
  ovarian and prostate arms were needed for.

## Round 7 — what the numbers are, 2026-09-17

| # | Comment | Status | What changed |
|---|---|---|---|
| 7.1 | 这个 retained / reject 是啥 (pointing at 0.0664 / 0.0600) | done | A real labelling fault, and worse than one column: **the word "retained" was doing three jobs on that one page**, two of them in the same 0–1 range. The top table's 0.0664 is a signature *score*, the lower-left table's 0.186 is a *fraction of cells*, and the lower-right table is the variance of the binary *label*. Renamed to `retained 组得分` / `rejected 组得分`, `被留下的细胞比例` with `rejected 比例` / `retained 比例`, and `留下 / 拒绝 这个标签的方差来源`. |
| 7.2 | p 值大家都懂,关键是啥怎么计算的 | done | Fair — the glossary was explaining a p value to people who know what one is, and leaving out the part they cannot see. Replaced with the procedure: UCell scores each cell on that dataset's own Top-50 metastasis signature (0–1); each patient gives one median per group and their difference; **the unit of inference is the patient, not the cell** — 29 paired differences in ovarian, two-sided paired Wilcoxon. The page also says why the unit matters: with cells as the unit that is 84,465 observations and any trivial difference returns an unusable p, whereas patients ask whether it is consistent between people. The same summary is now the first item under the table. |

### A factual error round 1 caught

I had written "三个数据集里方向都是反的" on the overview and in the page-4
heading. That is wrong, and comment 1.2 is what exposed it. From the table:

| Dataset | Difference | p |
|---|---:|---:|
| Ovarian GSE180661 | **+0.0063** (direction correct) | 1.9e-8 |
| Head and neck GSE181919 | −0.0099 (reversed) | 0.375 |
| Colorectal GSE225857 | 0.0000 (no signal) | 1.000 |

Only head and neck is reversed. The accurate statement, now used in both
places, is that none of the three supports the direction: the only significant
one is an artefact, and the other two did not replicate — one reversed, one
flat.

---

## Standing decisions

These came out of earlier rounds and should not be re-litigated without a
reason.

- **Androgen response agrees.** The pseudobulk DESeq2 plus preranked GSEA is the
  instrument that call rests on. A per-cell score over a focused gene set is the
  weaker measurement and is not plotted against it.
- **Contrast is primary retained vs rejected.** Every axis names its baseline.
  The metastasis appears only as a reference series on the same baseline, never
  as the thing being contrasted.
- **Arms are always named.** The free rule retains 1.6% in prostate and the
  capped rule 15–18%, so a retained percentage without its arm is not
  checkable. The prostate embedding is from the capped arm and says so.
- **No `set -u`** in any shell script here; an explicit path check replaces it.
- Shell commands emit no Chinese.

## Open

- MT% upper tail in prostate needs a TACC query; only per-pair medians were
  collected, so it cannot be checked locally.
- Ovarian DEG to be rerun on the depth-corrected gate with cell cycle regressed
  out.
