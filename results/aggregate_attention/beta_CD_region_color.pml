# beta-CD - Region-colour attention map
# grey=Backbone  orange=Cavity  green=Primary rim  blue=Secondary rim
# Run from results/aggregate_attention/:  pymol beta_CD_region_color.pml

load beta_CD.pdb, cd_host

# Define region colours
set_color col_backbone, [0.533, 0.533, 0.533]
set_color col_cavity, [0.878, 0.482, 0.224]
set_color col_primary, [0.298, 0.686, 0.314]
set_color col_secondary, [0.357, 0.608, 0.835]

color col_backbone, cd_host

color col_cavity, (cd_host and index 3+8+15+22+29+36+43+50+56+60+64+68+72+76)
color col_primary, (cd_host and index 1+2+9+10+16+17+23+24+30+31+37+38+44+45)
color col_secondary, (cd_host and index 51+52+53+54+55+57+58+59+61+62+63+65+66+67+69+70+71+73+74+75+77)

hide everything
show surface, cd_host
show sticks, cd_host
set transparency, 0.30, cd_host
orient
zoom cd_host, 5
bg_color white

# Export 1920x1080 PNG
png beta_CD_region_color.png, width=1920, height=1080, dpi=300, ray=1