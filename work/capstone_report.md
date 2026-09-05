# Capstone Report — Lane 4: CTR / Engagement Opportunity Scoring

- **Author:** Jana Moustafa
- **Lane:** Lane 4 — CTR / Engagement Opportunity Scoring
- **Repo:** [https://github.com/JanaMoustafa/FlyRank-ML](https://github.com/JanaMoustafa/FlyRank-ML)
- **Deployed Research Paper:** [https://janamoustafa.github.io/FlyRank-ML/](https://janamoustafa.github.io/FlyRank-ML/)
- **Date:** March 2026

---

## 1. Problem framing & FlyRank Case Study

### FlyRank Operational Context
FlyRank builds content as infrastructure: it autonomously researches keyword clusters, writes articles, publishes directly into client CMS instances, and continuously monitors search telemetry. However, once content ranks on Google SERPs, it quietly decays: rankings fluctuate, competitor snippets evolve, and click-through rates slip. 

In production, FlyRank's platform has historically surfaced hand-written heuristic flags (such as static `needs-attention` tags, `quick-win` labels, or blanket CTR threshold rules). While easy to implement, fixed rules suffer from severe systemic flaws:
1. **False Alarms on Deep Ranks:** They recommend title rewrites on deep-tier pages (e.g. position 18) where sub-1% CTR is the natural baseline for that ranking tier.
2. **Missed High-Visibility Deficits:** They miss high-ranking URLs (e.g. Page 1 positions 4–8) where a 2% CTR represents an enormous click deficit compared to tier peers.
3. **Capacity Constraints:** An editorial squad has capacity to rewrite only 50 to 100 pages per month. Misallocating these hours wastes copywriter effort while leaving high-impression click leaks unresolved.

This project supports the decision of **prioritizing URLs for title, snippet, and content engagement optimization**.
- **Unit of Analysis:** Individual URL aggregated over a 90-day observation window.
- **Output:** Calibrated opportunity probability score ($0.0 \to 1.0$) and a ranked operational action queue with explicit reason codes (`RC_HIGH_IMP_LOW_CTR`, `RC_THIN_CONTENT_UNDERPERFORM`, `RC_HIGH_BOUNCE_POOR_TIME`, `RC_STALE_POSITION_DROP`).
- **Human Action:** Copywriters and SEO leads inspect the top-ranked queue, evaluate SERP search intent, and perform targeted title/meta-description rewrites or content expansions.
- **Cost of False Positives:** Expending 2–4 hours of copywriter time rewriting an already well-performing title snippet that risks keyword ranking volatility.
- **Cost of False Negatives:** Leaving high-impression search visibility dormant when simple metadata adjustments could unlock thousands of missed monthly visits.

ML solves this by identifying multi-dimensional non-linear patterns across positional tiers, search volumes, dwell metrics, and freshness that simple 1D sorting cannot capture.

---

## 2. Data safety

### Inclusions & Release
- **Release:** FlyRank ML Internship Dataset Release (March 2026 panel, anonymized enterprise subset).
- **Working Filter:**
  - `avg_position > 0` (valid Google SERP presence)
  - `impressions_90d >= 100` (sufficient statistical sample to assess click viability)
  - `position_tier != "no_data"` (assigned to `top_3`, `page_1`, `striking`, `page_3_5`, or `deep`)
- Resulting working set: **22,006 rows across 30 distinct enterprise clients**.

### Exclusions & Public Safety
- **Deliberately Excluded:** Raw search queries, customer brand domains, private URLs, user identifiers, and raw IP addresses.
- **Leakage Safeguards:** Strictly excluded forward-looking columns (`trend_direction`, `trend_pct`, `impressions_prev_30d`, `clicks_prev_30d`) and direct target reconstruction proxies (`clicks_90d`, contemporaneous `ctr`).
- **Grouping Only:** Pseudonymous `client_id` and `content_id` are strictly reserved for cross-validation grouping and deduplication, never passed as model features.

---

## 3. Baseline

### Rule-Based Tier Expectation Baseline
Our transparent baseline represents standard industry practice: computing the expected median CTR for each position tier ($\text{CTR}_{\text{median}}(\text{tier})$) and calculating the shortfall weighted by search volume:
$$\text{Score}_{\text{baseline}} = \max(0, \text{CTR}_{\text{median}} - \text{CTR}) \times \log(1 + \text{impressions})$$

- **Why it's a fair comparison:** It reflects the best heuristic rule an experienced SEO analyst would construct using spreadsheet formulas.
- **Baseline GroupKFold Performance:**
  - ROC-AUC: 0.988 (when evaluated with direct CTR access, as it mirrors the label definition)
  - When evaluated strictly on non-CTR metadata features, linear models achieve ROC-AUC of 0.868, setting the honest benchmark to beat.

---

## 4. Model / analysis

### Target Label Definition
$$\text{is\_low\_ctr\_for\_tier} = \mathbb{I}(\text{CTR} < \text{Tier\_P25}(\text{Position Tier}))$$
Base rate across the dataset: **10.01%** (2,202 positive opportunity rows out of 22,006).

### Clean Feature Set (5 Features Knowable at Decision Moment)
1. `log_imp_month`: $\log(1 + \text{impressions}_{90\text{d}})$ — search volume scale.
2. `avg_pos_month`: Average Google SERP ranking position.
3. `ga4_eng_rate`: GA4 on-page user engagement rate.
4. `pct_days_with_impressions`: Proportion of days with active search impressions over the 90-day panel.
5. `days_since_update`: Days elapsed since the URL's last substantive content update.

### Algorithm
We deploy a **Random Forest Classifier** (`n_estimators=150`, `max_depth=8`, `min_samples_leaf=20`, `class_weight="balanced"`, `random_state=42`). The tree ensemble captures non-linear interactions between SERP visibility tiers and on-page dwell times without parametric assumptions.

---

## 5. Evaluation

### GroupKFold Cross-Validation (5 Folds by `client_id`)
Splitting by `client_id` prevents data leakage across client domains, ensuring our reported generalization reflects performance on completely unseen websites.

### Model vs Baseline Comparison Table (5-Fold GroupKFold)

| Model | Group CV ROC-AUC | P@20 | P@50 | P@100 | Lift over Base Rate |
|---|---|---|---|---|---|
| **Random Guessing (Base Rate)** | 0.500 | 10.0% | 10.0% | 10.0% | 1.00× |
| **Logistic Regression** | 0.868 | 45.0% | 46.0% | 45.0% | 4.50× |
| **Decision Tree (depth 3)** | 0.909 | 50.0% | 52.0% | 51.0% | 5.10× |
| **Random Forest (Lane 4 Model)** | **0.922** | **70.0%** | **68.0%** | **65.6%** | **6.56×** |

*Key finding:* In the top-100 ranked candidates, the Random Forest achieves **65.6% precision**, representing an estimated **6.56× lift** over random editorial selection (+20.6 percentage points over the linear benchmark).

### Error Analysis & Concrete Failure Modes
- **False Negatives:** High-ranking URLs (average position 1.2–3.5) with large search impressions (5,000+) but modest CTR where engagement metrics are high (>0.60). The model treats the strong engagement as an indicator of page health, failing to recognize that title snippet copy fails to hook the searcher on the SERP.
- **False Positives:** Pages with high search impressions and modest CTR where brand dominance or navigation queries suppress non-primary link clicks.

---

## 6. Interpretation

### Feature Importance (Permutation Drop in ROC-AUC)
1. `avg_pos_month` (-0.084): Ranking position governs the expected CTR curve; deviation from expected tier behavior is the strongest opportunity indicator.
2. `log_imp_month` (-0.042): Impression volume separates high-confidence click deficits from low-volume noise.
3. `ga4_eng_rate` (-0.028): High engagement rate signals that content quality is sound and the deficit lies strictly in SERP presentation.
4. `days_since_update` (-0.016): Stale content exhibits progressive CTR decay as competitor snippets modernize.
5. `pct_days_with_impressions` (-0.011): Regularity of search visibility establishes ranking stability.

---

## 7. Recommendation (Action Playbook)

### Ranked Action Types
1. `CTR_TITLE_REWRITE` (`RC_HIGH_IMP_LOW_CTR`): Top-3 and Page-1 pages with severe CTR deficits. Intervention: Update meta title, add structured schema, clarify value proposition in snippet.
2. `EXPAND_THIN_CONTENT` (`RC_THIN_CONTENT_UNDERPERFORM`): Striking-distance pages with <600 words and low engagement. Intervention: Expand content depth, address secondary user queries.
3. `ENGAGEMENT_RESTRUCTURE` (`RC_HIGH_BOUNCE_POOR_TIME`): High impressions but poor dwell time. Intervention: Add above-the-fold table of contents, multimedia, and clear callouts.
4. `MONITOR_DECAY` (`RC_STALE_POSITION_DROP`): Content >180 days old showing nascent rank slippage.
5. `NO_ACTION` (`RC_HEALTHY_OR_LOW_PRIORITY`): Healthy pages requiring no editorial spend.

### Strict No-Go List for Automation
- ❌ **NEVER** auto-publish title or snippet modifications without editorial review.
- ❌ **NEVER** modify top-quartile revenue URLs through automated triggers.
- ❌ **NEVER** mass-deindex or consolidate URLs based solely on algorithmic output.

### 4-Step Human Review Protocol
1. **SERP Intent Verification:** Confirm the ranking query intent matches the current landing page value proposition.
2. **Branded Query Audit:** Verify that low CTR is not an artifact of navigational branded competitor queries.
3. **Snippet Preview:** Ensure revised title tags fit within Google's 600px desktop display limit.
4. **Post-Publish Tracking:** Monitor CTR and rank position for 21 days post-deployment; revert if rank drops >2 positions.

---

## 8. Reproducibility

- **Code Repository:** [https://github.com/JanaMoustafa/FlyRank-ML](https://github.com/JanaMoustafa/FlyRank-ML)
- **Execution Script:** Run all cells in `work/notebooks/capstone.ipynb`
- **Environment:** Python 3.12, scikit-learn 1.4+, pandas 2.2+, numpy 1.26+, matplotlib 3.8+
- **Fixed Random Seed:** `RANDOM_SEED = 42`
- **Data Credit:** Built on the FlyRank ML Internship dataset ([https://flyrank.ai](https://flyrank.ai))

---

## 9. Week-8 Showcase: 5-Minute Demo Outline

Ready for the Week-8 showcase presentation:

- **Minute 1 — The Problem & Question:** FlyRank manages thousands of client URLs. Editorial teams can only rewrite 50–100/mo. Legacy hand-written heuristic flags (`needs-attention`) treat CTR uniformly, causing false alarms on deep tiers while missing Page 1 click deficits.
- **Minute 2 — The Method & Leak-Free Design:** 22,006 production rows across 30 domains; 5 clean features knowable at decision time; 5-fold GroupKFold CV strictly by client domain.
- **Minute 3 — One Chart (Precision@K):** Walk through `fig_precision_at_k.png` showing sustained 65.6% precision at top-100 queue depth vs 10.01% base rate.
- **Minute 4 — One Honest Result:** 0.922 ROC-AUC and 6.56× precision lift over base rate. Clear honest framing of error modes (high dwell masks title hook failures) and unobserved SERP features.
- **Minute 5 — One Recommendation:** Operational action queue (`CTR_TITLE_REWRITE`), the strict automation No-Go list, and the 4-step human review sign-off.

---

## 10. Shareable Cuts

### Cut 1: Social Post (Methodology & Finding)
> Can machine learning beat static SEO heuristics when deciding which web pages to optimize? 🔍
>
> Most SEO teams rely on hand-written rules (e.g., "CTR < 2% = rewrite title"). But on real SERPs, position 18 naturally gets <1% CTR, while position 4 with 2% CTR is suffering from a massive click deficit.
>
> Working with 22,006 production URLs across 30 enterprise domains from FlyRank's search telemetry, I built a regularized Random Forest opportunity model using 5 leakage-free behavioral, positional, and freshness signals.
>
> Under strict 5-fold GroupKFold cross-validation (evaluating exclusively on unseen client domains), the model achieves:  
> • 0.922 ROC-AUC (±0.035)  
> • 65.6% Precision@100 (a 6.56× lift over the 10.01% random base rate)  
> • +20.6 percentage points precision gain over linear benchmarks  
>
> The output directly feeds an operational content action playbook with clear reason codes, 4-step human review safeguards, and strict automation limits.
>
> Read the full research paper & action playbook: https://janamoustafa.github.io/FlyRank-ML/  
> Repo & code: https://github.com/JanaMoustafa/FlyRank-ML  
>
> #MachineLearning #DataScience #SEO #FlyRank #GroupedValidation #Python

### Cut 2: Employer-Facing Summary (3 Sentences)
> 1. **What I built:** I engineered and validated an end-to-end SEO CTR opportunity scoring engine and operational action playbook that prioritizes high-impact content interventions for enterprise marketing teams.
> 2. **On what data:** The model was trained on 22,006 multi-client URL performance records from FlyRank’s production search console and GA4 warehouse release, evaluated under strict 5-fold GroupKFold cross-validation to ensure reliable generalization across completely unseen client websites without data leakage.
> 3. **What it showed:** The regularized Random Forest model achieved a 0.922 GroupKFold ROC-AUC and 65.6% Precision@100 (a 6.56× lift over the 10.01% base rate, beating linear baselines by +20.6%), translating complex multi-signal telemetry into prioritized editorial actions with explicit human-review guardrails.
