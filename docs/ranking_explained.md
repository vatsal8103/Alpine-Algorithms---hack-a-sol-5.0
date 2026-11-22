# TRACE-X: Registry Matching, Ranking & Safe Abstention Explained

**For Hackasol 5.0 Judges & Forensic Examiners**

---

## 1. Multi-Criteria Evidence Combination

Vehicle shortlisting in TRACE-X uses a composite similarity function combining partial-plate positional evidence with physical attributes:

$$\text{Final Score} = \frac{w_{\text{plate}} \cdot S_{\text{plate}} + w_{\text{colour}} \cdot S_{\text{colour}} + w_{\text{body}} \cdot S_{\text{body}} + w_{\text{vis}} \cdot S_{\text{vis}}}{\sum w_k}$$

Default calibrated weights:
- **Partial Plate ($w_{\text{plate}} = 0.65$)**: Primary identification evidence.
- **Body Colour ($w_{\text{colour}} = 0.15$)**: Perceptual compatibility matrix based on CCTV lighting studies.
- **Body Type ($w_{\text{body}} = 0.12$)**: Classified probability distribution with partial-credit transitions.
- **Visual Appearance ($w_{\text{vis}} = 0.08$)**: Appearance embedding cosine similarity (optional).

If any sensor component is missing or unreliable, the system renormalizes remaining weights to sum to 1.0.

---

## 2. Positional Plate Scoring with Confusable Penalties

Unlike raw Levenshtein edit distance, TRACE-X positionally compares the fused consensus string against candidate registry plates:
1. **Exact Match**: $+1.0 \times \text{confidence}_k$
2. **Wildcard (`?`)**: Neutral $+0.95$ contribution (neither rewards nor heavily penalizes unknown positions).
3. **Optical Confusion Pairs**: Mismatch between $(O, 0)$, $(B, 8)$, $(I, 1)$, $(S, 5)$, $(Z, 2)$, $(D, 0)$, or $(G, 6)$ receives a discounted penalty ($-0.35$ instead of $-1.00$).
4. **Hard Contradiction**: Mismatch of an unambiguous, high-confidence character ($\ge 0.60$) applies a $-1.00$ penalty and trips an automatic abstention flag.

---

## 3. The 4-Way Safe Abstention Rule (No-Match Handling)

A forensic system must know when to say *"I don't know"*. TRACE-X outputs `NO_CONFIDENT_MATCH` under any of four conditions:

1. **Unreadable Plate**: If $\ge 70\%$ of character positions are `?`, the evidence is insufficient for candidate attribution.
2. **Below Absolute Threshold**: If the highest composite score is below `no_match_threshold` ($0.52$).
3. **High-Confidence Character Contradiction**: If candidate #1 disagrees with any verified character with confidence $\ge 0.60$.
4. **Narrow Margin Under Ambiguity**: If plate uncertainty is high ($>40\%$ `?`) and the score margin between Candidate #1 and Candidate #2 is below `margin_threshold` ($0.06$).

---

## 4. Soft Investigator Hints vs Hard Filters

When an investigator enters optional prior hints (e.g. *"Red SUV"*):
- Many legacy tools apply hard SQL filters (`WHERE colour = 'red' AND body = 'SUV'`).
- If an eyewitness misremembered or the vehicle was repainted, the true culprit is permanently erased from the shortlist.
- **TRACE-X Innovation**: Hints are Bayesian soft priors. They adjust attribute probabilities without filtering out candidates. If the camera clearly shows a silver vehicle, the report explicitly alerts the investigator of the discrepancy rather than crashing or returning an empty set.
