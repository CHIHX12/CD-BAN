# CD-BAN attention on crystal structure: gamma-CD
# Source: PDB 2ZYK chain E
# Color: blue=low attention, white=mid, red=high attention

load gamma_CD_crystal.pdb, cd_host

# Set all B-factors to 0 first
alter cd_host, b=0.0

# Map attention score to B-factor by atom name (role)
alter (cd_host and name C1), b=0.592256
alter (cd_host and name C2), b=0.028510
alter (cd_host and name C3), b=0.018074
alter (cd_host and name C4), b=0.437309
alter (cd_host and name C5), b=0.535173
alter (cd_host and name C6), b=0.268410
alter (cd_host and name O2), b=0.919903
alter (cd_host and name O3), b=0.958236
alter (cd_host and name O4), b=0.797255
alter (cd_host and name O5), b=0.935089
alter (cd_host and name O6), b=0.900957

# Color spectrum: blue(low) -> white(mid) -> red(high)
spectrum b, blue_white_red, cd_host, minimum=0, maximum=1

# Display settings
hide everything
show sticks, cd_host
show surface, cd_host
set transparency, 0.30, cd_host
set stick_radius, 0.15

# View
orient cd_host
zoom cd_host, 3
set bg_color, white

# Render 1920x1080
ray 1920, 1080
png gamma_CD_crystal_render.png, dpi=300

# Legend:
# blue  = low attention (C2, C3 backbone carbons)
# white = intermediate (C1, C4, C5, C6)
# red   = high attention (O5 ring-O, O2/O3 secondary-OH, Ob bridge-O)