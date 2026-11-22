# TRACE-X: Multi-Frame Evidential Fusion Explained

**For Hackasol 5.0 Judges & Forensic Evaluators**

---

## 1. The Core Problem: Why Single-Frame OCR Fails in CCTV
Single-frame CCTV plate recognition fails due to:
- Motion blur stretching characters horizontally
- Low spatial resolution (sub-15px character heights)
- Compression blocking and sensor photon shot noise
- Partial occlusion from tow bars, dirt, or angle distortion

Conventional systems make a binary decision per frame or run library-based majority voting. When evidence is noisy, majority voting either produces erratic characters or silently hallucinates wrong letters.

---

## 2. Our Original Multi-Frame Evidential Fusion Algorithm

TRACE-X replaces majority voting with a 5-step probabilistic Bayesian consensus fuser:

### Step 1: Quality-Aware Frame Weighting
Each video frame $i$ receives a normalized weight $w_i$ based on three independent quality signals:
$$w_i \propto (\text{sharpness}_i)^{\alpha} \cdot (\text{plate\_area}_i)^{\beta} \cdot (\text{mean\_ocr\_conf}_i)^{\gamma}$$
where default exponents are $\alpha = 0.35$, $\beta = 0.25$, $\gamma = 0.40$. Frames with high edge definition and large bounding box area contribute proportionally more evidence.

### Step 2: Needleman-Wunsch String Alignment with Confusable Costs
Different frames may have dropped or inserted characters. TRACE-X aligns candidate strings using dynamic programming with reduced substitution penalties ($0.35$ vs $1.00$) for known optical confusions:
- `O` $\leftrightarrow$ `0`
- `B` $\leftrightarrow$ `8`
- `I` $\leftrightarrow$ `1`
- `S` $\leftrightarrow$ `5`
- `Z` $\leftrightarrow$ `2`
- `D` $\leftrightarrow$ `0`
- `G` $\leftrightarrow$ `6`

### Step 3: Evidential Posterior Accumulation
For each character slot $k$, evidence is accumulated across all frames and their top-3 CTC alternatives:
$$\text{Evidence}(c, k) = \sum_{i=1}^{N} w_i \cdot \left[ P_i(c, k) + \lambda \sum_{\text{alt}} P_i(\text{alt}, k) \right]$$
where $\lambda = 0.30$ weights alternative predictions.

### Step 4: Indian Registration Positional Priors
Indian license plates follow strict positional syntax:
- Position 0–1: State code ($\text{Alphabet}$)
- Position 2–3: District RTO ($\text{Digits}$)
- Position 4–5: Series letters ($\text{Alphabet}$)
- Position 6–9: Vehicle number ($\text{Digits}$)

Characters that violate positional class expectations receive a syntax penalty ($0.60\times$) rather than silent replacement.

### Step 5: Honest Decision Rule & Abstention
A character $c^*$ is accepted into the final plate **if and only if**:
1. **Posterior Probability**: $P(c^* \mid \text{evidence}) \ge \theta_{\text{accept}}$ (default $0.50$)
2. **Margin over Runner-Up**: $P(c^* \mid \text{evidence}) - P(c_{\text{runner-up}} \mid \text{evidence}) \ge \Delta_{\text{margin}}$ (default $0.12$)

If either condition fails, the character is set to `?`.

---

## 3. Worked Numeric Example

Consider position 7 (expected digit) across 3 frames:
- **Frame 1** ($w_1 = 0.50$): predicts `6` ($p=0.42$), alternative `8` ($p=0.38$)
- **Frame 2** ($w_2 = 0.30$): predicts `8` ($p=0.45$), alternative `6` ($p=0.35$)
- **Frame 3** ($w_3 = 0.20$): predicts `8` ($p=0.40$), alternative `0` ($p=0.20$)

Accumulated evidence:
- Evidence for `8`: $0.50 \cdot (0.38 \cdot 0.30) + 0.30 \cdot 0.45 + 0.20 \cdot 0.40 = 0.057 + 0.135 + 0.080 = 0.272$
- Evidence for `6`: $0.50 \cdot 0.42 + 0.30 \cdot (0.35 \cdot 0.30) = 0.210 + 0.0315 = 0.2415$

Posterior for `8`: $0.53$, Posterior for `6`: $0.47$.
Margin $= 0.53 - 0.47 = 0.06$.

Since margin ($0.06$) is **below** $\Delta_{\text{margin}} = 0.12$, TRACE-X outputs `?`.
Explanation recorded:
`"Position 7: '8' (p=0.53) vs runner-up '6' (p=0.47) - margin 0.06 < threshold 0.12"`

**Result**: Instead of falsely committing to `8` or `6`, the system records `?` and lets the registry matching engine search for candidates matching both variants.

---

## 4. Why Marking "?" is Safer Than Guessing
1. **Prevents Investigatory Blind Spots**: Falsely outputting `8` when the true digit is `6` will cause the registry query to completely miss the culprit's vehicle.
2. **Transparent Accountability**: Every `?` is backed by a full vote log showing all frames and alternative probabilities for judicial review.
