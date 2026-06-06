# CD-BAN Attention Visualization
# Naproxen inside alpha-CD
# Colour = normalised BANLayer attention (blue=low, red=high)

load /home/nibiohnproj9/cycheng/cd_gnn_training/CD_BAN/results/pymol/alpha_CD.pdb, cd_host
load /home/nibiohnproj9/cycheng/cd_gnn_training/CD_BAN/results/pymol/naproxen.pdb, guest

alter cd_host, b=0.0

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
alter (cd_host and index 36), b=0.891719
alter (cd_host and index 37), b=0.995201
alter (cd_host and index 38), b=0.900827
alter (cd_host and index 39), b=0.987696
alter (cd_host and index 40), b=0.900827
alter (cd_host and index 41), b=0.987696
alter (cd_host and index 42), b=0.891719
alter (cd_host and index 43), b=0.995201
alter (cd_host and index 44), b=0.900827
alter (cd_host and index 45), b=0.987696
alter (cd_host and index 46), b=0.891719
alter (cd_host and index 47), b=0.995201
alter (cd_host and index 48), b=0.900827
alter (cd_host and index 49), b=0.987696
alter (cd_host and index 50), b=0.891719
alter (cd_host and index 51), b=0.995201
alter (cd_host and index 52), b=0.900827
alter (cd_host and index 53), b=0.987696
alter (cd_host and index 54), b=0.891719
alter (cd_host and index 55), b=0.995201

spectrum b, blue_white_red, cd_host, minimum=0, maximum=1

hide everything
show surface, cd_host
show sticks, cd_host
set transparency, 0.35, cd_host

show sticks, guest
color yellow, guest
set stick_radius, 0.15, guest

orient
zoom cd_host, 5
set bg_color, white


# png results/pymol/alpha_CD_naproxen.png, width=1200, height=900, dpi=300