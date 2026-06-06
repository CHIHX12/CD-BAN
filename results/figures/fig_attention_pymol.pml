# CD-BAN Attention Coloring Script
# Host: β-CD
# Guest attention mapped to B-factor, then colored by spectrum
# Usage: run this .pml file in PyMOL after loading your CD structure
#   load your_cd_structure.pdb, cd_host
#   @results/figures/fig_attention_pymol.pml

# Reset B-factors to 0
alter cd_host, b=0

# Set B-factor per atom index (0-based)
alter (cd_host and index 1), b=0.965443
alter (cd_host and index 2), b=0.864816
alter (cd_host and index 3), b=0.924352
alter (cd_host and index 4), b=0.997220
alter (cd_host and index 5), b=0.999600
alter (cd_host and index 6), b=1.000000
alter (cd_host and index 7), b=0.960602
alter (cd_host and index 8), b=0.924353
alter (cd_host and index 9), b=0.864818
alter (cd_host and index 10), b=0.965443
alter (cd_host and index 11), b=0.997220
alter (cd_host and index 12), b=0.999600
alter (cd_host and index 13), b=1.000000
alter (cd_host and index 14), b=0.960602
alter (cd_host and index 15), b=0.924353
alter (cd_host and index 16), b=0.864818
alter (cd_host and index 17), b=0.965443
alter (cd_host and index 18), b=0.997220
alter (cd_host and index 19), b=0.999600
alter (cd_host and index 20), b=1.000000
alter (cd_host and index 21), b=0.960602
alter (cd_host and index 22), b=0.924353
alter (cd_host and index 23), b=0.864818
alter (cd_host and index 24), b=0.965443
alter (cd_host and index 25), b=0.997220
alter (cd_host and index 26), b=0.999600
alter (cd_host and index 27), b=1.000000
alter (cd_host and index 28), b=0.960602
alter (cd_host and index 29), b=0.924353
alter (cd_host and index 30), b=0.864818
alter (cd_host and index 31), b=0.965443
alter (cd_host and index 32), b=0.997220
alter (cd_host and index 33), b=0.999600
alter (cd_host and index 34), b=1.000000
alter (cd_host and index 35), b=0.960602
alter (cd_host and index 36), b=0.924353
alter (cd_host and index 37), b=0.864818
alter (cd_host and index 38), b=0.965443
alter (cd_host and index 39), b=0.997220
alter (cd_host and index 40), b=0.999600
alter (cd_host and index 41), b=1.000000
alter (cd_host and index 42), b=0.960602
alter (cd_host and index 43), b=0.924353
alter (cd_host and index 44), b=0.864818
alter (cd_host and index 45), b=0.965443
alter (cd_host and index 46), b=0.997220
alter (cd_host and index 47), b=0.999600
alter (cd_host and index 48), b=1.000000
alter (cd_host and index 49), b=0.960602
alter (cd_host and index 50), b=0.891719
alter (cd_host and index 51), b=0.995201
alter (cd_host and index 52), b=0.900827
alter (cd_host and index 53), b=0.987696
alter (cd_host and index 54), b=0.900827
alter (cd_host and index 55), b=0.987696
alter (cd_host and index 56), b=0.891719
alter (cd_host and index 57), b=0.995201
alter (cd_host and index 58), b=0.900827
alter (cd_host and index 59), b=0.987696
alter (cd_host and index 60), b=0.891719
alter (cd_host and index 61), b=0.995201
alter (cd_host and index 62), b=0.900827
alter (cd_host and index 63), b=0.987696
alter (cd_host and index 64), b=0.891719
alter (cd_host and index 65), b=0.995201
alter (cd_host and index 66), b=0.900827
alter (cd_host and index 67), b=0.987696
alter (cd_host and index 68), b=0.891719
alter (cd_host and index 69), b=0.995201
alter (cd_host and index 70), b=0.900827
alter (cd_host and index 71), b=0.987696
alter (cd_host and index 72), b=0.891719
alter (cd_host and index 73), b=0.995201
alter (cd_host and index 74), b=0.900827
alter (cd_host and index 75), b=0.987696
alter (cd_host and index 76), b=0.891719
alter (cd_host and index 77), b=0.995201

# Color by B-factor (attention score)
spectrum b, blue_white_red, cd_host, minimum=0, maximum=1

# Optional: show surface + sticks
show surface, cd_host
show sticks, cd_host
set transparency, 0.3, cd_host

# Save session
# save results/figures/fig_attention_pymol_session.pse