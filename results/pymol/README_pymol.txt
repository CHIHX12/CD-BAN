CD-BAN PyMOL Visualization Files
==================================

Guest molecule: Naproxen  (CC(c1ccc2cccc(OC)c2c1)C(=O)O)
Four cyclodextrin types: alpha-CD, beta-CD, gamma-CD, HP-beta-CD

File listing
------------
naproxen.pdb           - drug 3D structure (RDKit ETKDGv3)
alpha_CD.pdb           - alpha-CD 3D structure
beta_CD.pdb            - beta-CD 3D structure
gamma_CD.pdb           - gamma-CD 3D structure
HP_beta_CD.pdb         - HP-beta-CD 3D structure

*_attention.csv        - per-atom attention scores for each CD

alpha_CD_naproxen.pml  - PyMOL script (alpha-CD + Naproxen)
beta_CD_naproxen.pml   - PyMOL script (beta-CD + Naproxen)
gamma_CD_naproxen.pml  - PyMOL script (gamma-CD + Naproxen)
HP_beta_CD_naproxen.pml- PyMOL script (HP-beta-CD + Naproxen)

comparison_heatmap.png/svg - side-by-side attention heatmaps for 4 CD types

PyMOL usage
-----------
1. Install PyMOL:
   conda install -c conda-forge pymol-open-source

2. Open PyMOL and run:
   @<full_path>/beta_CD_naproxen.pml

3. Color legend:
   blue   = low attention (CD atoms less attended by the model)
   white  = intermediate attention
   red    = high attention (CD atoms associated with binding by the model)
   yellow = Naproxen (drug guest)

Note:
   PDB structures are generated from SMILES by RDKit (ETKDGv3 force-field
   optimization). The drug position is not a real docking pose; it is
   illustrative only. For precise docking positions, use AutoDock Vina or
   similar tools.
