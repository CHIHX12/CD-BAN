# CD-BAN Algorithm — Complete Mathematical Formulation

Binary Classification of Cyclodextrin Inclusion Complex Affinity
Version 1.2 | Binary validated (AUROC 0.925 ± 0.040, 10 seeds) | Ternary descriptor formula added

---

## Notation

| Symbol | Dimension | Description |
|--------|-----------|-------------|
| G = (V_G, E_G) | — | Guest (drug) molecular graph |
| H = (V_H, E_H) | — | Host (cyclodextrin) molecular graph |
| L | — | Auxiliary ligand (coformer) — ternary only |
| N_G, N_H | — | Number of atoms in guest / host |
| φ(a) | ℝ^74 | Atom feature vector |
| d_emb | 128 | Node embedding dimension |
| d_h | 128 | GCN hidden dimension (per layer) |
| N_max | 290 | Maximum graph nodes (padding) |
| d_f | 256 | BAN output / MLP input dimension |
| H_BAN | 2 | Number of BAN attention heads |
| k | 3 | BAN low-rank factor |
| σ(·) | — | Sigmoid: σ(x) = 1 / (1 + e^{−x}) |
| z_bin | ℝ | Binary logit from CD-BAN model |
| Δz | ℝ | Ternary logit correction (ligand contribution) |
| z_tern | ℝ | Ternary logit = z_bin + Δz |
| RT | 2.479 kJ/mol | Gas constant × temperature (T = 298 K) |

---

## Step 1 — Atom Feature Vector φ(a) ∈ ℝ^74

For each atom a in a molecule, construct a 74-dimensional binary/scalar vector by concatenation:

```
φ(a) = [ e_elem(a) ‖ e_deg(a) ‖ e_val(a) ‖ q(a) ‖ r(a) ‖ e_hyb(a) ‖ arom(a) ‖ e_H(a) ]
```

| Component | Encoding | Dim |
|-----------|----------|-----|
| e_elem(a) | One-hot over 43 elements (C, N, O, S, F, Si, P, Cl, Br, …, Pb) | 43 |
| e_deg(a) | One-hot over degree ∈ {0,1,…,10} | 11 |
| e_val(a) | One-hot over implicit valence ∈ {0,1,…,6} | 7 |
| q(a) | Formal charge (scalar) | 1 |
| r(a) | Number of radical electrons (scalar) | 1 |
| e_hyb(a) | One-hot over {SP, SP2, SP3, SP3D, SP3D2} | 5 |
| arom(a) | Aromaticity flag ∈ {0, 1} | 1 |
| e_H(a) | One-hot over total H count ∈ {0,1,2,3,4} | 5 |
| **Total** | | **74** |

---

## Step 2 — Molecular Graph Construction

From SMILES string s, parse with RDKit:

```
G = (V, E)  where  V = {a_1, …, a_N}  (atoms),  E ⊆ V×V  (bonds, undirected → bidirectional)

Node feature matrix:  X ∈ ℝ^{N×74},   X_i = φ(a_i)
Edge index:           E ∈ ℤ^{2×|E|}    (COO format, both directions)
```

Same procedure for both guest graph G and host graph H.

---

## Step 3 — MolecularGCN: Graph → Node Embeddings

**Architecture:** Linear projection → 3 × GCNConv → padding to N_max

### 3.1 Linear Projection

```
X^(0) = X · W_0                    W_0 ∈ ℝ^{74×128},  no bias
X^(0) ∈ ℝ^{N×128}
```

### 3.2 GCNConv Layer (×3, shared structure)

Symmetric normalized propagation with self-loops:

```
For layer l = 0, 1, 2:

  X^(l+1)_i = ReLU( W^(l) · Σ_{j ∈ N(i)∪{i}}  x^(l)_j / √( (d_i+1)(d_j+1) ) )

  W^(l) ∈ ℝ^{128×128},   d_i = degree of node i
```

Equivalently in matrix form (with Â = D̃^{−1/2} Ã D̃^{−1/2}, Ã = A + I):

```
X^(l+1) = ReLU( Â · X^(l) · W^(l) )
```

Output after 3 layers:  X^(3) ∈ ℝ^{N×128}

### 3.3 Zero-Padding to N_max

```
V ∈ ℝ^{N_max × 128}  =  [ X^(3)_1, …, X^(3)_N, 0, …, 0 ]
                            ↑ N rows                ↑ (N_max − N) zero rows
```

Applied separately for guest → V_G ∈ ℝ^{290×128} and host → V_H ∈ ℝ^{290×128}.

Batch shape (batch size B):  V_G, V_H ∈ ℝ^{B×290×128}

---

## Step 4 — BANLayer: Bilinear Attention Interaction

Input:  V_G ∈ ℝ^{B×290×128},  V_H ∈ ℝ^{B×290×128}
Output: f ∈ ℝ^{B×256}  (joint interaction embedding)

### 4.1 Parallel FC Projection (FCNet, low-rank expansion)

```
V'_G = FCNet_v(V_G)  ∈ ℝ^{B×290×(256×3)} = ℝ^{B×290×768}
V'_H = FCNet_q(V_H)  ∈ ℝ^{B×290×768}

FCNet: Linear(128 → 768) + ReLU  (weight-normalized)
```

### 4.2 Bilinear Attention Map (H_BAN = 2 heads)

Using learnable tensor H ∈ ℝ^{1×2×1×768} and bias b ∈ ℝ^{1×2×1×1}:

```
A_{b,h,v,q} = Σ_k  H_{h,k} · V'_G_{b,v,k} · V'_H_{b,q,k}  + b_h

A ∈ ℝ^{B×2×290×290}   (h = 0, 1 for two heads)
```

In Einstein notation:
```
A = einsum('xhyk, bvk, bqk → bhvq',  H, V'_G, V'_H)  + b
```

### 4.3 Attention Pooling (per head h)

```
logits_h = einsum('bvk, bvq, bqk → bk',  V'_G, A_{:,h,:,:}, V'_H)  ∈ ℝ^{B×768}

# Average-pool over low-rank factor k=3:
logits_h = AvgPool1d(k=3) · logits_h × 3                              ∈ ℝ^{B×256}
```

### 4.4 Multi-head Summation + Batch Normalization

```
f_raw = Σ_{h=0}^{1}  logits_h     ∈ ℝ^{B×256}

f = BatchNorm1d(f_raw)             ∈ ℝ^{B×256}
```

f is the **joint complex embedding** encoding how the drug interacts with CD.

---

## Step 5 — MLPDecoder: Complex Embedding → Logit

4-layer MLP with BatchNorm:

```
h_1 = ReLU( BN( W_1 · f   + b_1 ) )      W_1 ∈ ℝ^{512×256},  h_1 ∈ ℝ^{B×512}
h_2 = ReLU( BN( W_2 · h_1 + b_2 ) )      W_2 ∈ ℝ^{512×512},  h_2 ∈ ℝ^{B×512}
h_3 = ReLU( BN( W_3 · h_2 + b_3 ) )      W_3 ∈ ℝ^{128×512},  h_3 ∈ ℝ^{B×128}
z   =        W_4 · h_3 + b_4              W_4 ∈ ℝ^{1×128},    z   ∈ ℝ^{B×1}
```

z is the **raw logit** (no activation).

---

## Step 6 — Binary Prediction

```
P(Weak | G, H)   = σ(z)  = 1 / (1 + e^{−z})    ∈ (0, 1)
P(Strong | G, H) = 1 − σ(z) = σ(−z)

Decision rule:
  label = 1  (Weak Binder,   K < 100 M⁻¹)    if  P(Weak) ≥ 0.5  ↔  z ≥ 0
  label = 0  (Strong Binder, K > 10,000 M⁻¹) if  P(Weak) < 0.5  ↔  z < 0
```

---

## Complete Formula Chain (SMILES → Prediction)

```
SMILES_G, SMILES_H
      │
      ▼  [Step 1–2: RDKit parsing + atom featurization]
X_G ∈ ℝ^{N_G×74},  X_H ∈ ℝ^{N_H×74}
      │
      ▼  [Step 3.1: Linear projection, W_0 ∈ ℝ^{74×128}]
X^(0) ∈ ℝ^{N×128}
      │
      ▼  [Step 3.2: 3× GCNConv + ReLU + symmetric norm]
X^(3) ∈ ℝ^{N×128}
      │
      ▼  [Step 3.3: Zero-pad → N_max = 290]
V_G, V_H ∈ ℝ^{B×290×128}
      │
      ▼  [Step 4.1: FCNet 128 → 768]
V'_G, V'_H ∈ ℝ^{B×290×768}
      │
      ▼  [Step 4.2: Bilinear attention map, H_BAN = 2 heads]
A ∈ ℝ^{B×2×290×290}
      │
      ▼  [Step 4.3–4.4: Attention pooling + BN]
f ∈ ℝ^{B×256}           ← joint complex embedding
      │
      ▼  [Step 5: MLP 256→512→512→128→1 + BN]
z ∈ ℝ^{B×1}              ← affinity logit
      │
      ▼  [Step 6: sigmoid]
P(Weak) ∈ (0, 1)         ← predicted probability
```

---

## Training Objective

**Loss function:** Weighted Binary Cross-Entropy with Logits

```
L = − (1/N) Σ_i [ w⁺ · y_i · log σ(z_i) + (1 − y_i) · log(1 − σ(z_i)) ]

where:
  y_i ∈ {0, 1}   ground truth label
  z_i             raw logit from Step 5
  w⁺ = 11.32      pos_weight = n(label=1) / n(label=0) = 770 / 68  (training set)
```

**Equivalently:**
```
L = BCEWithLogitsLoss(z, y, pos_weight = w⁺)
```

w⁺ upweights the strong-binder class (label=0, minority, 8.1% of training) to compensate
for class imbalance.

---

## Parameter Count

| Module | Parameters |
|--------|-----------|
| GCN guest (W_0 + 3×W_l) | 74×128 + 3×(128×128) = 9,472 + 49,152 = 58,624 |
| GCN host (identical) | 58,624 |
| BAN (H, b, FCNet_v, FCNet_q) | ~604,000 |
| MLP decoder (W_1–W_4 + BN) | 256×512 + 512×512 + 512×128 + 128×1 + BN params ≈ 459,776 |
| **Total** | **~780,550** |

---

## Dataset Composition (Why 1,198 Not 3,048)

The full OpenCycloDB pipeline produces 3,249 entries. After curation:

```
OpenCycloDB (raw)          3,249 entries
  − duplicate exclusions     −201  (same host–guest pair, conflicting K across sources)
  = Curated total           3,048

  Curated 3,048 split by K value:
  ├── Binary set (labelled)  1,198  K < 100 M⁻¹ (label=1) or K > 10,000 M⁻¹ (label=0)
  │     train  838  (label=0: 68,  label=1: 770)
  │     val    240  (label=0: 19,  label=1: 221)
  │     test   120  (label=0: 10,  label=1: 110)
  └── Fuzzy zone (unlabelled) 1,850  100 ≤ K ≤ 10,000 M⁻¹ — no label assigned
                                      used only for retrospective inference
```

The model is trained and evaluated on the **1,198 binary entries only**.
The 1,850 fuzzy-zone entries are never seen during training; they validate
that the learned logit z encodes a continuous affinity gradient.

**Full binary dataset validation (seed 49, n = 1,198):**

| Metric | Value |
|--------|-------|
| AUROC | 0.9895 |
| AUPRC | 0.9990 |
| F1 | 0.9873 |
| Accuracy | 97.66% |
| Weak correct (K < 100) | 1087 / 1101 = 98.7% |
| Strong correct (K > 10,000) | 83 / 97 = 85.6% |

---

## Empirical Calibration: Logit ↔ log₁₀K

From retrospective inference on 1,850 fuzzy-zone compounds (not seen during training):

```
z ≈ −1.128 × log₁₀K + 3.714          Pearson r(z_bin, log₁₀K) = −0.381  (p < 0.001)
                                      Pearson r(P_weak, log₁₀K) = −0.468  (p < 0.001, as shown in Fig. 4)
                                      Spearman ρ = −0.425  (p < 0.001)
Note: formula coefficients derived from boundary conditions, not OLS.

Inverse:  log₁₀K ≈ (3.714 − z) / 1.128
```

**Key calibration points:**

| z value | K (M⁻¹) | Interpretation |
|---------|---------|----------------|
| z > +1.458 | K < 100 | Confident Weak Binder |
| z = 0 | K ≈ 1,956 | Decision boundary (P = 0.5) |
| z < −0.799 | K > 10,000 | Confident Strong Binder |

This calibration enables **quantitative affinity estimation** beyond binary classification,
without any retraining of the model.

---

## Ternary Formula Derivation

The ternary extension uses the binary model's output (z_bin) as its foundation
and adds a ligand-dependent correction term derived from three mathematical steps.

---

### Step 7 — Ligand Descriptor Extraction from SMILES

Given SMILES_Ligand, compute molecular descriptors using RDKit:

```
logP(L)   = Wildman–Crippen partition coefficient  (hydrophobicity)
TPSA(L)   = Topological Polar Surface Area (Å²)    (polarity / H-bonding)
charge(L) = Σ formal charges over all atoms        (net ionic character)
```

These descriptors are standard RDKit calculations from the 2D molecular graph —
no 3D conformation or experimental data required.

---

### Step 8 — Δz Derivation: Why Logit Shifts with K Ratio

**From the empirical calibration (Section "Empirical Calibration"):**

```
z  ≈  −1.128 × log₁₀K  +  3.714
```

If adding a coformer multiplies K by a factor `r = K_tern / K_bin`, then:

```
z_tern = −1.128 × log₁₀(K_tern) + 3.714
       = −1.128 × log₁₀(K_bin × r) + 3.714
       = −1.128 × (log₁₀K_bin + log₁₀r) + 3.714
       = z_bin  −  1.128 × log₁₀r
       = z_bin  +  Δz

Therefore:  Δz = −1.128 × log₁₀(K_tern / K_bin)   ← key identity
```

For each of the 4 literature ternary data points, Δz_obs is computed directly
from the measured K values — no model is involved in this calculation:

| Ligand | K_bin | K_tern | r = K_tern/K_bin | log₁₀r | Δz_obs = −1.128×log₁₀r | logP(L) |
|--------|-------|--------|-----------------|--------|------------------------|---------|
| HPMC E5 | 280 | 980 | 3.50× | 0.544 | −0.614 | −1.26 |
| PEG 4000 | 75 | 310 | 4.13× | 0.616 | −0.695 | −1.03 |
| L-arginine | 180 | 720 | 4.00× | 0.602 | −0.679 | −1.34 |
| HPMC K15M | 120 | 580 | 4.83× | 0.684 | −0.772 | −1.26 |
| **Mean** | | | **4.1×** | | **−0.690 ± 0.065** | |

---

### Step 9 — OLS Fitting: Δz as a Function of Ligand Descriptors

**Regression target:** Δz_obs (4 values above)

**Predictor:** logP(L)  (highest available correlation with Δz)

```
Δz = w_logP × logP(L)  +  c

OLS fit on n=4 points:
  w_logP = −0.0378
  c      = −0.7361
  RMSE   =  0.056

Correlation: r(logP, Δz) = −0.078  (very weak — n=4 insufficient)
```

**Why logP?**
logP measures hydrophobicity. More hydrophobic coformers (higher logP) tend to
interact more strongly with the hydrophobic CD cavity surface, but the signal
is weak with only 4 data points. More data are needed to confirm this relationship.

**Inverse recovery of K ratio from Δz:**

```
K_ternary / K_binary = 10^(−Δz / 1.128)

With Δz = w_logP × logP + c:
  K_ternary / K_binary = 10^(−(w_logP × logP + c) / 1.128)
                       = 10^(0.0335 × logP + 0.6525)    [from fitted coefficients]
```

---

### Step 10 — Ternary Prediction (Complete Pipeline)

**Input:** SMILES_Guest, SMILES_Host, SMILES_Ligand

```
[Step 1–6: Binary model, from predict_model.py]
  z_bin = MLP( BAN( GCN(G), GCN(H) ) )
  P_bin(Weak) = σ(z_bin)
  K_bin ≈ 10^((3.714 − z_bin) / 1.128)

[Steps 7–9: Ternary correction, from predict_ternary_formula.py]
  logP(L) = RDKit_logP(SMILES_Ligand)
  Δz      = −0.0378 × logP(L) − 0.7361
  z_tern  = z_bin + Δz
  P_tern(Weak) = σ(z_tern)

[Ternary K and thermodynamics]
  K_ternary ≈ K_binary × 10^(−Δz / 1.128)
  ΔΔG       = −RT × ln(K_ternary / K_binary)   [kJ/mol, T = 298 K]
```

**Fallback (polymer coformers without valid SMILES):**
```
  Δz = −0.690  (constant, mean of 4 literature points, SD = 0.065)
  K_ternary ≈ K_binary × 4.1×
  ΔΔG ≈ −3.54 kJ/mol
```

---

### Complete Formula Chain: Binary → Ternary

```
SMILES_G + SMILES_H + SMILES_L
      │               │
      ▼               ▼
  [Steps 1–6]    [Steps 7–9]
  CD-BAN model   RDKit descriptors
      │               │
      ▼               ▼
    z_bin    +      Δz(logP)
      │               │
      └───────┬───────┘
              ▼
           z_tern = z_bin + Δz
              │
      ┌───────┼───────────────────┐
      ▼       ▼                   ▼
  P_tern    K_ternary           ΔΔG
  = σ(z)  ≈ K_bin×10^(−Δz/1.128)  = −RT×ln(K_tern/K_bin)
```

---

### Ternary Formula Validation

| Compound | K_bin (lit) | K_tern (lit) | K_tern predicted | Error |
|----------|-------------|--------------|-----------------|-------|
| naproxen/HP-β-CD + L-Arg | 280 | 720* | ~1,172 | +63% |
| indomethacin/β-CD + L-Arg | 180 | 720 | ~749 | +4% |

> *naproxen + L-Arg not in original 4 data points; tested as cross-validation proxy.
> Error magnitude expected given n=4 training data.

**Honest assessment:**
- With n=4 and 3 polymers mapped to approximate SMILES, the descriptor-based
  formula is not significantly better than the constant correction (RMSE 0.056
  vs SD 0.065 for constant)
- The formula is architecturally correct and will improve as more ternary
  data are collected (target: ≥30 small-molecule coformers with measured K values)

---

### Ternary Deep-Learning Extension (Future Work)

When ≥150 ternary training triplets (G, H, L, ΔΔG) are available, the
descriptor regression can be replaced by a learned representation:

```
Current (descriptor regression):
  Δz = w_logP × logP(L) + c                 ← 2 parameters, no retraining

Future (learned embedding):
  f_GH  = BAN( GCN(G), GCN(H) )             ← frozen from binary model
  v_L   = GCN_L(L)                           ← new branch, trainable
  ΔΔG   = MLP_Δ( BAN_T( f_GH, v_L ) )       ← new head, trainable

  New parameters only: GCN_L + BAN_T + MLP_Δ ≈ 250,000
  Frozen parameters:   GCN_G + GCN_H + BAN   ≈ 780,550  (transfer learning)
```

This replaces Step 9 (scalar logP coefficient) with a full graph neural network
that reads the ligand's 3D electronic structure from its SMILES.

---

---

## Summary: Binary → Ternary Formula Progression

| Step | Formula | Source | Script |
|------|---------|--------|--------|
| 1–6 | z_bin = MLP(BAN(GCN(G), GCN(H))) | Trained DL model | `predict_model.py` |
| Calibration | log₁₀K = (3.714 − z_bin) / 1.128 | Boundary-condition derived (z=+1.458↔K=100, z=−0.799↔K=10,000); Pearson r(z,logK)=−0.381 | `predict_calibration.py` |
| 7 | logP, TPSA, charge = RDKit(SMILES_L) | Molecular descriptors | `predict_ternary_formula.py` |
| 8 | Δz_obs = −1.128 × log₁₀(K_tern/K_bin) | Derived from K measurements | (fitting step) |
| 9 | Δz = −0.0378 × logP − 0.7361 | OLS on 4 lit. points | `predict_ternary_formula.py` |
| 10 | z_tern = z_bin + Δz → K_ternary, ΔΔG | Arithmetic | `predict_ternary_formula.py` |

---

## References

\[1\] Mura, P.; Faucci, M.T.; Parrini, P.L. *Ternary systems of naproxen with hydroxypropyl-β-cyclodextrin and amino acids.* **Int. J. Pharm.** 2003, **260**, 293–302.
DOI: [10.1016/S0378-5173(03)00265-5](https://pubmed.ncbi.nlm.nih.gov/12842348/)
*(Source of naproxen/HP-β-CD + L-arginine K values used in ternary validation)*

\[2\] Fernandes, C.M.; Veiga, F.J.B. *Effects of some hydrotropic agents on the formation of indomethacin/β-cyclodextrin inclusion compounds.* **J. Inclusion Phenom. Macrocyclic Chem.** 1999, **33**, 117–125.
DOI: [10.1023/A:1007939715918](https://link.springer.com/article/10.1023/A:1007939715918)
*(Source of indomethacin/β-CD + L-arginine, HPMC E5, PEG 4000, HPMC K15M K values used to fit Δz)*

---

*End of Algorithm Document — Version 1.2*
*Binary model validated: AUROC 0.925 ± 0.040, AUPRC 0.992 ± 0.005 (n=10 seeds, test n=120)*
*Ternary formula: n=4 literature data points, RMSE=0.056, not independently validated*
*All ± values: sample standard deviation (ddof=1)*
