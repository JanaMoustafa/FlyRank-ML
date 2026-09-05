# Capstone Report — Lane 4: CTR / Engagement Opportunity Scoring

- **Author:** Jana Moustafa
- **Lane:** Lane 4 — CTR / Engagement Opportunity Scoring
- **Repo:** [https://github.com/JanaMoustafa/FlyRank-ML](https://github.com/JanaMoustafa/FlyRank-ML)
- **Deployed Research Paper:** [https://janamoustafa.github.io/FlyRank-ML/](https://janamoustafa.github.io/FlyRank-ML/)
- **Date:** March 2026

---

## 1. Problem framing

Enterprise SEO teams manage catalogs of tens of thousands of URLs across complex multi-intent domains. Editorial and optimization resources are finite: an editorial squad typically has capacity to audit and refresh only 50 to 100 pages per month. Without data-driven prioritization, teams rely on generic traffic rankings or ad-hoc requests, frequently wasting hours rewriting pages that already perform at their natural ceiling while ignoring high-impression URLs suffering from severe snippet deficits.

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
- **Leakage Safeguards:** Strictly excluded forward-looking columns (`trend_direction`, `trend_pct`, `impressions_prev_30d`, `clicks_prev_30d`) and direct target reconstruction proxies (`clicks_90d`).
- **Grouping Only:** Pseudonymous `client_id` and `content_id` are strictly reserved for cross-validation grouping and deduplication, never passed as model features.

---

## 3. Baseline

### Rule-Based Tier Expectation Baseline
Our transparent baseline represents standard industry practice: computing the expected median CTR for each position tier ($\text{CTR}_{\text{median}}(\text{tier})$) and calculating the shortfall weighted by search volume:
$$\text{Score}_{\text{baseline}} = \max(0, \text{CTR}_{\text{median}} - \text{CTR}) \times \log(1 + \text{impressions})$$

- **Why it's a fair comparison:** It reflects the best heuristic rule an experienced SEO analyst would construct using spreadsheet formulas.
- **Baseline GroupKFold Performance:**
  - ROC-AUC: 0.988 (when evaluated with direct CTR access)
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
| **Random Guessing (Base Rate)** | 0.500 | 10.0% | 10.0% | 10.0% | 1.0× |
| **Logistic Regression** | 0.868 | 45.0% | 46.0% | 45.0% | 4.5× |
| **Decision Tree (depth 3)** | 0.909 | 50.0% | 52.0% | 51.0% | 5.1× |
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
