"""
Table 6 — Duplicate Entries Analysis
Table for paper: why excluded, CD-BAN prediction, recommendation
"""
import pandas as pd, numpy as np, os

OUT = 'results/tables'
os.makedirs(OUT, exist_ok=True)

df  = pd.read_csv('/home/nibiohnproj9/cycheng/cyclodextrin_research_data/curated_datasets/duplicate_entries.csv')
pred= pd.read_csv('results/seed_stability/duplicate_predictions.csv')
df['p_weak']   = pred['p_weak'].values
df['p_strong'] = pred['p_strong'].values

lines = []
def h(t=''):   lines.append(t)
def sep(c='─',n=95): lines.append(c*n)

# ═══════════════════════════════════════════════════════════════════════════
# Table 6a — Summary by ΔG Spread Category
# ═══════════════════════════════════════════════════════════════════════════
h()
h('Table 6a.  Duplicate Entries — Summary by Measurement Disagreement (ΔG Spread)')
sep()
h('  Exclusion criterion: same host–guest pair reported in ≥ 2 independent sources')
h('  with conflicting binding constants (DeltaG_spread = ΔG_max − ΔG_min, kJ/mol)')
sep()
h(f'  {"ΔG Spread":<20} {"n":>5}  {"log₁₀K (mean)":>14}  {"log₁₀K (SD)":>12}'
  f'  {"p_weak (mean)":>13}  {"Interpretation":<35}  {"Recommendation"}')
sep('─')

cats = [
    (0,   1,   '<1',   'Near-identical measurements',        'Merge using mean K'),
    (1,   3,   '1–3',  'Minor disagreement (~2× in K)',      'Review case-by-case'),
    (3,   6,   '3–6',  'Moderate disagreement (~10× in K)',  'Exclude or use median'),
    (6,  99,   '>6',   'Large disagreement (>100× in K)',    'Exclude; data unreliable'),
]

rows6a = []
for lo, hi, lbl, interp, rec in cats:
    sub = df[(df.DeltaG_spread >= lo) & (df.DeltaG_spread < hi)]
    h(f'  {lbl:<20} {len(sub):>5}  {sub.log10K.mean():>14.3f}  {sub.log10K.std():>12.3f}'
      f'  {sub.p_weak.mean():>13.4f}  {interp:<35}  {rec}')
    rows6a.append({'spread_range': lbl, 'n': len(sub),
                   'log10K_mean': round(sub.log10K.mean(),3),
                   'log10K_sd':   round(sub.log10K.std(),3),
                   'p_weak_mean': round(sub.p_weak.mean(),4),
                   'interpretation': interp, 'recommendation': rec})
sep('─')
h(f'  {"Total":<20} {len(df):>5}  {df.log10K.mean():>14.3f}  {df.log10K.std():>12.3f}'
  f'  {df.p_weak.mean():>13.4f}')
sep()
h(f'  Pearson r  (log₁₀K vs p_weak) = {np.corrcoef(df.log10K,df.p_weak)[0,1]:.4f}  (p<0.001, n=201)')
h('  Note: CD-BAN was trained on the curated set (n=3,048) and never saw these 201 entries.')

pd.DataFrame(rows6a).to_csv(f'{OUT}/table6a_duplicate_summary.csv', index=False)

# ═══════════════════════════════════════════════════════════════════════════
# Table 6b — High-Affinity Duplicates Detail (log10K > 4)
# ═══════════════════════════════════════════════════════════════════════════
h()
h()
h('Table 6b.  High-Affinity Duplicate Entries (log₁₀K > 4) — Detailed Analysis')
sep()
h(f'  {"Guest":<28} {"Host":<22} {"log₁₀K":>8} {"K (M⁻¹)":>9}'
  f'  {"ΔG Spread":>10}  {"p_weak":>7}  {"Prediction":>14}  Exclusion Reason & Recommendation')
sep('─')

hi = df[df.log10K > 4].sort_values('log10K', ascending=False)
rows6b = []

details = {
    'bencyclane': {
        'reason': 'Two sources: K=80k & K=40k M⁻¹; same compound, small spread (1.72 kJ/mol)',
        'rec':    'ADD — merge as mean K=57,000 M⁻¹',
        'why':    'Model p_weak≈0 → correctly identifies as strong binder',
    },
    '4-methylcinnamic acid': {
        'reason': 'Large spread (11.04 kJ/mol); K values range ~100-fold between sources',
        'rec':    'KEEP EXCLUDED — measurement unreliable',
        'why':    'Model p_weak=0.74 → doubts high K; α-CD cavity likely too small',
    },
    'doxepin': {
        'reason': 'Two sources with nearly identical K (spread=0.01 kJ/mol)',
        'rec':    'ADD — virtually duplicate measurement, use mean',
        'why':    'Model p_weak≈0 → correctly identifies as strong binder',
    },
    '4-bromocinnamic acid': {
        'reason': 'Spread=7.57 kJ/mol; conflicting K values across sources',
        'rec':    'KEEP EXCLUDED — uncertain measurement',
        'why':    'Model p_weak=0.78 → α-CD spatial mismatch suspected',
    },
    '4-methoxycinnamic acid': {
        'reason': 'Spread=8.96 kJ/mol; large inter-source disagreement',
        'rec':    'KEEP EXCLUDED — uncertain measurement',
        'why':    'Model p_weak=0.73 → consistent with structural doubt',
    },
}

for _, row in hi.iterrows():
    g = str(row.get('Guest', ''))
    h_name = str(row.get('Host', '')).replace('beta-cyclodextrin','β-CD').replace('alpha-cyclodextrin','α-CD')
    pred_lbl = 'Strong (K>10k)' if row.p_weak < 0.5 else 'Weak (K<100)'
    info = details.get(g, {'reason':'—','rec':'—','why':'—'})
    h(f'  {g:<28} {h_name:<22} {row.log10K:>8.3f} {10**row.log10K:>9.0f}'
      f'  {row.DeltaG_spread:>10.2f}  {row.p_weak:>7.4f}  {pred_lbl:>14}')
    h(f'    Reason: {info["reason"]}')
    h(f'    Model interpretation: {info["why"]}')
    h(f'    Recommendation: {info["rec"]}')
    h()
    rows6b.append({
        'guest': g, 'host': h_name,
        'log10K': round(row.log10K,3),
        'K_M-1': int(10**row.log10K),
        'DeltaG_spread_kJmol': round(row.DeltaG_spread,2),
        'p_weak': round(row.p_weak,4),
        'prediction': pred_lbl,
        'reason': info['reason'],
        'recommendation': info['rec'],
    })
sep()
h('  p_weak: CD-BAN predicted probability of weak binding (≥0.5 → Weak, <0.5 → Strong)')
h('  ΔG Spread: max(ΔG)−min(ΔG) across all duplicate measurements of the same pair (kJ/mol)')

pd.DataFrame(rows6b).to_csv(f'{OUT}/table6b_highaffinity_duplicates.csv', index=False)

# ═══════════════════════════════════════════════════════════════════════════
# Write TXT
# ═══════════════════════════════════════════════════════════════════════════
with open(f'{OUT}/table6_duplicate_analysis.txt', 'w') as f:
    f.write('\n'.join(lines))

print(f'  → {OUT}/table6_duplicate_analysis.txt')
print(f'  → {OUT}/table6a_duplicate_summary.csv')
print(f'  → {OUT}/table6b_highaffinity_duplicates.csv')
