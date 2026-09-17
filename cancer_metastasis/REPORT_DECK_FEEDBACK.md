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

### A factual error this round caught

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
