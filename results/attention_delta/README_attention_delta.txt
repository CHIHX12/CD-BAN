CD-BAN — Guest-Atom Attention Signature (attention_delta.png)
=============================================================

WHAT THIS FIGURE ANSWERS
------------------------
"Which atoms of the drug does the model pay attention to, and does that
pattern differ between STRONG binders (label 0) and WEAK binders (label 1)?"

The figure has three horizontal-bar panels that share the same Y axis:

    [ Weak (label 1) ]   [ Strong (label 0) ]   [ Delta = Strong - Weak ]


THE AXES
--------
Y axis (all three panels, same order):
    10 categories of GUEST (drug) atoms.
      - Aromatic C, Aliphatic C, N, O, S, Halogen   (by element / type)
      - H-bond donor, H-bond acceptor               (by polarity)
      - Ring atom, Non-ring atom                    (by topology)
    NOTE: the categories OVERLAP (an -OH oxygen counts under "O", "H-bond
    donor" and "H-bond acceptor"), so the bars do NOT sum to 1 across all 10.

X axis:
    Panel 1 (Weak) and Panel 2 (Strong):
        "mean attention fraction" = on average, what fraction of the model's
        attention on the drug lands on atoms of that category.
        0.0 = that atom type is never attended; 0.5 = half of the attention
        mass sits on that atom type.

    Panel 3 (Delta = Strong - Weak):
        the strong-panel value minus the weak-panel value, per category.
        RED bar pointing RIGHT  (positive) = that atom type gets MORE
            attention in STRONG binders.
        BLUE bar pointing LEFT  (negative) = MORE attention in WEAK binders.


HOW THE NUMBERS ARE MADE
------------------------
1. Run CD-BAN (seed 49) on all 1,198 labelled drug-CD pairs.
2. Take the BANLayer attention and turn it into a distribution over the
   drug's atoms (softmax) -> each atom gets an "attention share".
3. Add up the shares by atom category -> one attention fraction per category.
4. Average over the 1,101 weak pairs  -> Panel 1
   average over the    97 strong pairs -> Panel 2
   subtract                            -> Panel 3 (Delta)


HOW TO READ IT (KEY FINDING)
----------------------------
Look at Panel 3 (Delta). The largest bars are the most discriminative:

    Aliphatic C   delta = +0.19   (RED  -> strong binders)
    Ring atom     delta = +0.10   (RED  -> strong binders)
    Aromatic C    delta = -0.11   (BLUE -> weak binders)
    Non-ring atom delta = -0.10   (BLUE -> weak binders)
    H-bond accept delta = -0.08   (BLUE -> weak binders)
    H-bond donor  delta = -0.05   (BLUE -> weak binders)

So:
    STRONG binders -> attention concentrates on ALIPHATIC CARBON + RING atoms
                      (the hydrophobic part that inserts into the CD cavity)
    WEAK binders   -> attention on AROMATIC C and POLAR / H-bonding atoms
                      (groups that prefer water over the cavity)

An attention-only logistic "formula" built from these 10 features separates
strong vs weak at AUROC = 0.776, balanced accuracy = 0.72 (5-fold CV).


PHYSICAL MEANING
----------------
Cyclodextrin inclusion is driven by the HYDROPHOBIC EFFECT. The model was
never told this, yet its attention encodes it: strong complexes are the ones
where the drug presents an aliphatic / ring hydrophobic surface to the cavity.
This is an interpretable-AI confirmation that CD-BAN learned the correct
physical driver of host-guest binding.


CAVEATS
-------
- Categories overlap (see note above); do not expect bars to sum to 1.
- Class imbalance: 97 strong vs 1,101 weak. Class means are robust, but the
  strong panel is averaged over fewer molecules.
- This is a SUPERVISED read-out (uses labels to average per class). The same
  signal is NOT strong enough for unsupervised KPCA clustering (see
  ../attention_kpca/), i.e. it is a thin but real discriminative direction.


FILES
-----
attention_delta.png / .svg   the 3-panel figure (600 dpi)
attention_delta.csv          per-category weak / strong / delta / logit weight
run_attention_delta.py       the script that produces everything
run_log.txt                  console log of the run
