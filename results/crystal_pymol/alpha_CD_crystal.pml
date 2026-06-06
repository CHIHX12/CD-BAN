# CD-BAN attention on crystal structure: alpha-CD
# Source: PDB 4FEM chain B
# Color: blue=low attention, white=mid, red=high attention

load alpha_CD_crystal.pdb, cd_host

# Set all B-factors to 0 first
alter cd_host, b=0.0

# Map attention score to B-factor by atom name (role)
alter (cd_host and name C1), b=0.647809
alter (cd_host and name C2), b=0.068386
alter (cd_host and name C3), b=0.042831
alter (cd_host and name C4), b=0.461269
alter (cd_host and name C5), b=0.543828
alter (cd_host and name C6), b=0.275558
alter (cd_host and name O2), b=0.907795
alter (cd_host and name O3), b=0.937240
alter (cd_host and name O4), b=0.809322
alter (cd_host and name O5), b=0.941388
alter (cd_host and name O6), b=0.856491

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
png alpha_CD_crystal_render.png, dpi=300

# Legend:
# blue  = low attention (C2, C3 backbone carbons)
# white = intermediate (C1, C4, C5, C6)
# red   = high attention (O5 ring-O, O2/O3 secondary-OH, Ob bridge-O)