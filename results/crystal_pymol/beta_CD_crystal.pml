# CD-BAN attention on crystal structure: beta-CD
# Source: PDB 1DMB chain B
# Color: blue=low attention, white=mid, red=high attention

load beta_CD_crystal.pdb, cd_host

# Set all B-factors to 0 first
alter cd_host, b=0.0

# Map attention score to B-factor by atom name (role)
alter (cd_host and name C1), b=0.711887
alter (cd_host and name C2), b=0.094073
alter (cd_host and name C3), b=0.063645
alter (cd_host and name C4), b=0.519604
alter (cd_host and name C5), b=0.535723
alter (cd_host and name C6), b=0.102718
alter (cd_host and name O2), b=0.883639
alter (cd_host and name O3), b=0.918040
alter (cd_host and name O4), b=0.876813
alter (cd_host and name O5), b=0.982428
alter (cd_host and name O6), b=0.663423

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
png beta_CD_crystal_render.png, dpi=300

# Legend:
# blue  = low attention (C2, C3 backbone carbons)
# white = intermediate (C1, C4, C5, C6)
# red   = high attention (O5 ring-O, O2/O3 secondary-OH, Ob bridge-O)