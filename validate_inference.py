"""
End-to-end inference validation on literature-known compounds.
Verifies the complete algorithm chain: SMILES -> P(Weak Binder)
"""
import torch, yaml, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from models import CDBAN
from dataloader import smiles_to_pyg
from torch_geometric.data import Batch

with open('configs/CDBAN.yaml') as f:
    cfg = yaml.safe_load(f)

model = CDBAN(**cfg)
model.load_state_dict(torch.load('results/seed_49/best_model_epoch_69.pth', map_location='cpu'))
model.eval()

# ── Test compounds with known literature K values ──
# Format: (guest_name, guest_SMILES, host_name, host_SMILES, K_lit M⁻¹, expected_label)

BETA_CD = ('OC[C@H]1O[C@@H]2O[C@@H]3[C@@H](CO)O[C@H](O[C@@H]4[C@@H](CO)O[C@H](O'
           '[C@@H]5[C@@H](CO)O[C@H](O[C@@H]6[C@@H](CO)O[C@H](O[C@@H]7[C@@H](CO)'
           'O[C@H](O[C@@H]8[C@@H](CO)O[C@H](O[C@H]1[C@H](O)[C@H]2O)[C@H](O)[C@H]'
           '8O)[C@H](O)[C@H]7O)[C@H](O)[C@H]6O)[C@H](O)[C@H]5O)[C@H](O)[C@H]4O)'
           '[C@H](O)[C@H]3O')

ALPHA_CD = ('OC[C@H]1O[C@@H]2O[C@@H]3[C@@H](CO)O[C@H](O[C@@H]4[C@@H](CO)O[C@H](O'
            '[C@@H]5[C@@H](CO)O[C@H](O[C@@H]6[C@@H](CO)O[C@H](O[C@H]1[C@H](O)'
            '[C@H]2O)[C@H](O)[C@H]6O)[C@H](O)[C@H]5O)[C@H](O)[C@H]4O)[C@H](O)[C@H]3O')

tests = [
    # Known WEAK binders (K < 100 M⁻¹) — expect label=1, P(Weak)>=0.5
    ('ibuprofen',       'CC(C)Cc1ccc(cc1)C(C)C(=O)O',              'beta-CD',  BETA_CD,  320,   'Weak'),
    ('caffeine',        'Cn1cnc2c1c(=O)n(c(=O)n2C)C',              'beta-CD',  BETA_CD,  40,    'Weak'),
    ('acetic acid',     'CC(=O)O',                                   'alpha-CD', ALPHA_CD, 6,     'Weak'),
    ('paracetamol',     'CC(=O)Nc1ccc(O)cc1',                       'beta-CD',  BETA_CD,  25,    'Weak'),
    # Known STRONG binders (K > 10,000 M⁻¹) — expect label=0, P(Weak)<0.5
    ('testosterone',    'O=C1CC[C@@H]2[C@]3(C)CC[C@H]4C[C@@H](O)CC[C@]4(C)[C@@H]3CC[C@]12C',
                                                                     'beta-CD',  BETA_CD,  49000, 'Strong'),
    ('progesterone',    'CC(=O)[C@@H]1CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@H]3CC[C@]12C',
                                                                     'beta-CD',  BETA_CD,  45000, 'Strong'),
    ('estradiol',       'OC1=CC2=C(CC[C@@H]3[C@]2(CC[C@@H]3O)C)C=C1', 'beta-CD', BETA_CD, 32000, 'Strong'),
    ('cholesterol',     'CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C',
                                                                     'beta-CD',  BETA_CD,  19000, 'Strong'),
]

hdr = '{:<15s}  {:<9s}  {:>9s}  {:<8s}  {:>7s}  {:>8s}  {:<10s}  {}'
sep = '-' * 80
print(hdr.format('Guest', 'Host', 'K_lit', 'Expected', 'z', 'P(Weak)', 'Predicted', 'OK?'))
print(sep)

correct = 0
results = []
for g_name, g_smi, h_name, h_smi, k_lit, expected in tests:
    g = smiles_to_pyg(g_smi)
    h = smiles_to_pyg(h_smi)
    bg = Batch.from_data_list([g])
    bh = Batch.from_data_list([h])
    with torch.no_grad():
        _, _, score, _ = model(bg, bh, mode='eval')
    z = score.item()
    p = torch.sigmoid(score).item()
    pred = 'Weak' if p >= 0.5 else 'Strong'
    ok = 'V' if pred == expected else 'X'
    if pred == expected:
        correct += 1
    results.append((g_name, h_name, k_lit, expected, z, p, pred, ok))
    row = '{:<15s}  {:<9s}  {:>9,}  {:<8s}  {:>7.3f}  {:>8.4f}  {:<10s}  {}'
    print(row.format(g_name, h_name, k_lit, expected, z, p, pred, ok))

print(sep)
print('Accuracy: {}/{} = {:.0f}%'.format(correct, len(tests), correct / len(tests) * 100))
print()
print('Algorithm calibration:')
print('  z > +1.458  P(Weak) > 0.811  ->  Confident Weak   (K < 100 M-1)')
print('  z = 0       P(Weak) = 0.500  ->  Decision boundary (K ~ 1,956 M-1)')
print('  z < -0.799  P(Weak) < 0.310  ->  Confident Strong (K > 10,000 M-1)')
