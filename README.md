# CD-BAN: Cyclodextrin Bilinear Attention Network

Binary classification of drug–cyclodextrin inclusion complex affinity using GCN + BANLayer + MLP.

**Author:** Chih-Yang Cheng | **License:** MIT

---

## What it does

CD-BAN predicts whether a drug–cyclodextrin pair forms a **strong** (K > 10,000 M⁻¹) or **weak** (K < 100 M⁻¹) inclusion complex from SMILES strings alone, with no 3D geometry required.

| Label | Condition | Meaning |
|-------|-----------|---------|
| 0 | K > 10,000 M⁻¹ | Strong binder |
| 1 | K < 100 M⁻¹ | Weak binder |

---

## Key Results (10 seeds, test set n = 120)

| Metric | Mean ± SD |
|--------|-----------|
| AUROC | 0.925 ± 0.040 |
| AUPRC | 0.992 ± 0.005 |
| F1 | 0.957 ± 0.015 |
| Accuracy | 0.923 ± 0.025 |

---

## Installation

```bash
conda create -n cdban python=3.10
conda activate cdban
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric scikit-learn scipy matplotlib pandas pyyaml prettytable tqdm rdkit
```

---

## Training

```bash
# Single seed
python main.py --cfg configs/CDBAN.yaml --seed 42 --out results/seed_42

# All 10 seeds (sequential)
bash run_seeds.sh

# All 10 seeds (4-GPU parallel)
bash run_seeds_parallel.sh
```

---

## Prediction

Three scripts, run in order:

### Step 1 — Model inference (deep learning)
```bash
python predict_model.py \
  --guest "Cn1cnc2c1c(=O)n(c(=O)n2C)C" \
  --host  "OC[C@H]1O[C@@H]2..." \
  --name  "caffeine / beta-CD"
```
Output: `z_bin` (logit), `P(Weak)`, confidence label

### Step 2 — Estimate K (binary formula)
```bash
python predict_calibration.py --z 1.2308
```
Formula: `log₁₀K ≈ (3.714 − z_bin) / 1.128`

### Step 3 — Ternary K estimate (with coformer)
```bash
python predict_ternary_formula.py \
  --z 0.9380 \
  --ligand "N[C@@H](CCCNC(=N)N)C(=O)O" \
  --name "L-arginine"
```
Formula: `Δz = −0.0378 × logP(coformer) − 0.7361`

> Full mathematical derivation: [`ALGORITHM.md`](ALGORITHM.md)

---

## Architecture

```
Guest SMILES ──► MolecularGCN (3×GCNConv 128-128-128) ──►┐
                                                           ├──► BANLayer (2 heads) ──► MLP ──► logit
Host SMILES  ──► MolecularGCN (3×GCNConv 128-128-128) ──►┘
```

- Atom features: 74-dim (element, degree, valence, hybridization, aromaticity, H count)
- BANLayer: bilinear attention, adapted from DrugBAN
- Loss: BCEWithLogitsLoss, pos_weight = 11.32
- Parameters: 780,550

---

## Repository Structure

```
CD_BAN/
├── main.py                    # Training entry point
├── models.py                  # CDBAN, MolecularGCN, MLPDecoder
├── trainer.py                 # Training loop
├── dataloader.py              # Dataset and collate function
├── ban.py                     # BANLayer
├── utils.py                   # Utilities
├── predict_model.py           # Step 1: DL inference → z_bin
├── predict_calibration.py     # Step 2: binary K estimate
├── predict_ternary_formula.py # Step 3: ternary K estimate
├── predict_fuzzy.py           # Inference on fuzzy-zone compounds
├── predict_duplicates.py      # Inference on excluded duplicates
├── plot_figures.py            # Reproduce all publication figures
├── plot_attention.py          # BANLayer attention heatmap
├── generate_pymol_attention.py# PyMOL attention visualization
├── ALGORITHM.md               # Mathematical formulation
├── configs/CDBAN.yaml         # Hyperparameters
├── data/binary/               # train / val / test / fuzzy CSV files
└── results/                   # Model weights, figures, tables
```

---

## Dataset

| Split | N | Condition |
|-------|---|-----------|
| Train | 838 | K < 100 or K > 10,000 M⁻¹ |
| Val | 240 | same |
| Test | 120 | same |
| Fuzzy (inference only) | 1,850 | 100 ≤ K ≤ 10,000 M⁻¹ |

Source: OpenCycloDB (curated). Fuzzy-zone compounds are withheld from training and used only for retrospective inference.

---

## Citation

```bibtex
@article{cdban2026,
  title  = {CD-BAN: Binary Classification of Cyclodextrin Inclusion Complex Affinity via Bilinear Attention Network},
  author = {Cheng, Chih-Yang},
  year   = {2026},
  note   = {Preprint}
}
```
