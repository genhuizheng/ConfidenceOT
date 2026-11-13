# DEG / GSEA v3：同一 clade 内，RNA-OT 和 CRISPR-OT 偏好的细胞表达有没有差别（报告）

**日期：** 2026-11-13
**计划：** `PLAN_v3_deg.md`。2026-11-12 冻结；2026-11-13 补充了可执行的操作定义，写于读任何汇总之前。
**代码：**
- `100_deg_v3_groups.py`：分组；
- `60_deg_gsea.py`：DEG/GSEA，未改动；
- `103_ieg_sensitivity.py`：IEG 敏感性；
- `104_deg_v3_summary.py`：阅读规则；
- `105_deg_v3_figures.py`：图。

**结果位置：**
- 表：`results/rna_lineage_discordance_20261110/deg_v3/summary/`
- 图：`results/rna_lineage_discordance_20261110/figures/deg_v3/`

## 一句话结论

在同一个 CRISPR clade 内，RNA-OT 偏好的细胞和 CRISPR-OT 偏好的细胞确实有表达差异。但是：
- 这些差异大部分可以用"离 target 的 RNA 更近"（RNA 相似度 baseline）和测序深度来解释；
- 没有一个 hallmark 集达到预先定义的"一致"标准，在任何一类对比里都没有；
- 卡住的是随机分组对照：没有一个集合在至少一半的对比里超出随机分组的 NES 范围。

所以现在**不能**说两种 OT 的分歧对应某条通路。

## 做了什么

**范围：**
- KP 的 45 个 OT 单元，RNA 配置为 L5-best；
- 只用两个 gate 都保留的细胞；
- clade 用全部可用细胞得到的 A2 clade；state 用作者的 Cluster-Name。

**对比类型：** 共 257 个真实对比，覆盖 10 只小鼠，另有 705 个随机分组对比。

| 类型 | 分组依据 | 真实对比数 | 小鼠数 |
|---|---|---|---|
| A1：RNA-OT 偏好 vs CRISPR-OT 偏好（主对比，source 侧） | OT 差异 | 35 | 8 |
| A1：RNA-OT 偏好 vs 两者都支持 | OT 差异 | 22 | 8 |
| A1：CRISPR-OT 偏好 vs 两者都支持 | OT 差异 | 22 | 8 |
| A1 baseline：离 target 最近 vs 最远（RNA 余弦） | RNA 相似度 | 35 | 8 |
| A2：RNA-cross vs RNA-own（target 侧） | OT（RNA-OT 列质量） | 38 | 9 |
| A2 baseline：更接近别的 clade vs 自己的 clade | RNA 相似度 | 37 | 9 |
| B1：同一 clade、同一位点内最大的两个 state | RNA state | 29 | 9 |
| C1：同一 state、同一位点内最大的两个 clade | CRISPR edits | 39 | 10 |

**A1 的分组定义：**
- q_R(i) = 细胞 i 在本 clade 的 RNA-OT 质量中所占份额；q_C(i) 同理，用 CRISPR-OT；
- d = n × (q_R − q_C)；
- RNA-OT 偏好 = d 的上四分位，CRISPR-OT 偏好 = d 的下四分位。

**方法：**
- DEG 用 Wilcoxon，hallmark prerank 用全部 50 个集合；
- 两个版本：原始计数、P10 深度降采样计数；
- 主对比、B1、C1 各做 5 次随机分组；
- IEG 敏感性：从排序中去掉 23 个立即早期基因后重跑。

## 结果

### 1. 两个 OT 偏好的细胞，测序深度不同

见图 F7，数值为第一组与第二组的中位深度比。

| 对比 | 深度比中位数 | 说明 |
|---|---|---|
| RNA-OT 偏好 vs CRISPR-OT 偏好 | 1.40 | 86% 的对比里 RNA-OT 偏好组更深；检出基因数比 1.24 |
| CRISPR-OT 偏好 vs 两者都支持 | 0.78 | CRISPR-OT 偏好的细胞更浅 |
| RNA 相似度 baseline（离 target 最近 vs 最远） | 1.63 | |
| C1（clade X vs clade Y） | 0.97 | 不靠 RNA 选的对比，基本没有深度差 |

**解读：** RNA-OT 更偏好测序更深的细胞，这本身是 RNA cost 的性质：深度高的细胞在 RNA 空间里离 target 更近。两组的 edits 数一样（中位数都是 4）。

**降采样的作用：**
- 降采样后，A1 主对比 NES 的均值从 0.29 降到 0.09；
- NES 为正的比例从 59% 降到 48%；
- 但原始版和降采样版 NES 的 Spearman 仍有 0.92，说明方向大多保留。

### 2. A 类差异大部分是 RNA 相似度

见图 F4：每个对比的 50 个 NES，与同一 block 的 RNA 相似度 baseline 求 Spearman。

| 对比 | Spearman 中位数（原始 / 降采样） | top-100 DEG 同号重叠 |
|---|---|---|
| RNA-OT 偏好 vs CRISPR-OT 偏好 | 0.77 / 0.81 | 45% |
| RNA-cross vs RNA-own | 0.91 / 0.92 | 69% |
| RNA-OT 偏好 vs 两者都支持 | 0.44 / 0.50 | 11% |
| CRISPR-OT 偏好 vs 两者都支持 | −0.75 / −0.76 | 0% |

**解读：**
- RNA-OT 偏好哪些细胞，基本就是"哪些细胞的 RNA 离 target 更近"。这和计划里预期的选择偏倚一致：分组本身就是从 RNA 来的。
- CRISPR-OT 偏好的细胞方向正好相反，是 RNA 离 target 较远的细胞。它们不是从 RNA 选出来的，但通过谱系和 RNA 混杂在一起。

### 3. 符号一致的集合不少，但都过不了随机分组对照

见图 F2（各小鼠的 NES）和图 F5（超出随机范围的比例）。下表为包含 3724 的结果。

| 对比 | 原始和降采样都 ≥ 75% 小鼠同号 | 其中超出随机范围 ≥ 50% 的 | 一致 |
|---|---|---|---|
| RNA-OT 偏好 vs CRISPR-OT 偏好 | 17 / 50 | 0（最高 0.49，MYC targets V1） | 0 |
| RNA-cross vs RNA-own | 3 / 50 | 0 | 0 |
| RNA-OT 偏好 vs 两者都支持 | 5 / 50 | 没有随机分组，不评 | — |
| CRISPR-OT 偏好 vs 两者都支持 | 16 / 50 | 没有随机分组，不评 | — |

**主对比中同号的 17 个集合：**
- RNA-OT 偏好组更高：E2F、G2M、MYC V1/V2、OXPHOS、DNA repair、glycolysis、peroxisome、UPR、PI3K-AKT-mTOR、Notch、WNT、apical junction、allograft；
- RNA-OT 偏好组更低：hypoxia、IL6-JAK-STAT3、estrogen early。

**其中 14 个同时可以被 baseline 解释**，即 ≥ 50% 的对比里 baseline 同号且 |NES| ≥ 1。不被 baseline 解释的有：
- WNT β-catenin（被解释比例 0.14）；
- PI3K-AKT-mTOR（0.46）。

但这两个超出随机范围的比例分别只有 0.33 和 0.26。

**CRISPR-OT 偏好 vs 两者都支持：** 16 个集合在 ≥ 75% 的小鼠里为负，包括 MYC、E2F、OXPHOS、TGF-β、WNT 等，而且几乎都不被 baseline 解释。但这一类按计划没有随机分组，所以只报告方向，不判定一致。它的深度比是 0.78，降采样后方向保留。

**去掉 3724 后：** 主对比同号的集合从 17 个降到 7 个（DNA repair、Notch、OXPHOS、peroxisome、PI3K-AKT-mTOR、UPR、WNT），一致的仍然是 0 个。

### 4. 同一 clade 的例子

见图 F3。按预先定的规则选，即 focus 小鼠里细胞数最多的 A1 主对比：3724 family（primary→所有转移灶）、clade c45、2,816 个细胞，深度比 1.63。

- **RNA-OT 偏好组更高：** Vim、Lgals1、S100a4/a6/a10、Hmga1、Anxa2、Eno1、Mif（间充质、糖酵解）。
- **CRISPR-OT 偏好组更高：**
  - Malat1、Neat1、mt-Co1，这些常和细胞质量、核转录本比例有关；
  - Junb、Btg2、Nr4a1、Gadd45b、Cebpb，这些是立即早期 / 应激基因。
- CRISPR-OT 偏好组的深度明显更低。

**读法：** 在这个 clade 里，CRISPR-OT 偏好的细胞更像低深度、带应激信号的细胞。

一个可能的解释（未检验）：CRISPR-OT 的 cost 不含 RNA，所以细胞质量不影响它；RNA-OT 则偏向深度高、状态典型的细胞。

这是一个例子，不是统计结论。另外 3724 的 CRISPR 分辨率不足（按冻结规则未 resolved）。

### 5. IEG

- 每个对比 top-50 基因里 IEG 的比例，中位数在原始和降采样版都是 0–2%，最高 30%（A1 主对比）。
- 去掉 IEG 后重跑：与原 NES 的 Spearman 为 0.982（原始）/ 0.985（降采样）。
- **主对比：** 17 个同号集合中，有 14 个在去掉 IEG 后（原始和降采样两版）仍满足同号规则。不再满足的 3 个是 allograft rejection、hypoxia、IL6-JAK-STAT3。它们正好都是 CRISPR-OT 偏好组更高、偏炎症 / 应激的集合，说明其中一部分来自立即早期基因。
- **CRISPR-OT 偏好 vs 两者都支持：** 16 个同号集合中保留 15 个，失去 TGF-β。
- **RNA-cross vs RNA-own：** 3 个全部保留。
- 由于没有一致的集合，IEG 敏感性不改变结论。

### 6. B1 和 C1（不用 OT 的描述性对比）

见图 F6：|NES| 超出随机分组 |NES| 范围的对比比例。

**B1（同一 clade 内的两个 state）：**
- 每个集合的中位比例：原始 0.32，降采样 0.34；
- 原始和降采样都 ≥ 0.5 的集合：apoptosis、KRAS up、MYC V1；
- 去掉 3724 后，再加上 androgen response、cholesterol homeostasis。

**C1（同一 state 内的两个 clade）：**
- 中位比例为 0.26 / 0.26；
- 只有 apical surface 两版都 ≥ 0.5；
- 也就是说，同一 state 内不同 clade 之间的表达差异，大多不比随机分组大。

**B1 按 state 对分析：** 同一对 state 在 ≥ 2 只小鼠里出现的只有 5 对，各 2–3 只小鼠，例如 Endoderm-like vs Pre-EMT。方向一致的集合很多，但它们描述的是 state 之间的差别（state 本来就是从 RNA 定义的），只有 2–3 只小鼠。

## 能说的和不能说的

**能说：**
1. RNA-OT 在 clade 内偏好的细胞，主要是 RNA 离 target 更近、测序更深的细胞。和 RNA 相似度 baseline 的 NES 相关中位数为 0.77（target 侧 0.91）。
2. CRISPR-OT 偏好的细胞方向相反：更浅，多个增殖 / 代谢集合更低，在 ≥ 75% 的小鼠里同号，原始和降采样都一样。这一类没有随机分组对照，只是描述。
3. 不靠 RNA 选的 C1 对比（同一 state 内不同 clade）基本没有超出随机分组的通路差异。

**不能说：**
- 不能说两种 OT 的分歧对应某条 hallmark 通路，因为按预先规则一致的集合是 0 个；
- 不能把 3724 的 c45 例子推广；
- FDR 只是如实列出，不当作证据；每个位点只有一个肿瘤，细胞是伪重复。

## 图

| 文件 | 内容 |
|---|---|
| `F1_nes_heatmap_raw / _ds` | 全部 50 个集合 × 对比类型 × 小鼠的 NES（小鼠均值） |
| `F2_across_mice_<type>` | 每类对比各小鼠的 NES：实心 = 原始，空心 = 降采样，红 = 3724 |
| `F3_within_clade_heatmap` | 3724 c45 的细胞 × top 40 基因，带 group / state / 深度 / edits 轨道 |
| `F4_baseline_spearman` | 各对比与 RNA 相似度 baseline 的 NES 相关 |
| `F5_random_null_<type>` | 每个集合超出随机分组范围的对比比例 |
| `F6_bc_null_<type>` | B1 / C1 的 \|NES\| 超出随机范围的比例 |
| `F7_depth_ratio` | 每个对比两组的深度比 |

计划中的"flow + DEG 注释"图没有做：注释只放一致的集合，而一致的集合是 0 个。
