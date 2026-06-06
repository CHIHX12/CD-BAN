# HP-beta-CD - Region-colour attention map
# grey=Backbone  orange=Cavity  green=Primary rim  blue=Secondary rim
# Run from results/aggregate_attention/:  pymol HP_beta_CD_region_color.pml

load HP_beta_CD.pdb, cd_host

# Define region colours
set_color col_backbone, [0.533, 0.533, 0.533]
set_color col_cavity, [0.878, 0.482, 0.224]
set_color col_secondary, [0.357, 0.608, 0.835]
set_color col_HP, [0.612, 0.420, 0.620]

color col_backbone, cd_host

color col_cavity, (cd_host and index 7+12+19+26+33+40+47+54+59+67+75+83+91+99)
color col_secondary, (cd_host and index 13+14+15+20+21+22+27+28+29+34+35+36+41+42+43+48+49+50+55+56+57)
color col_HP, (cd_host and index 1+2+3+4+5+6+60+61+62+63+64+65+68+69+70+71+72+73+76+77+78+79+80+81+84+85+86+87+88+89+92+93+94+95+96+97+100+101+102+103+104+105)

hide everything
show surface, cd_host
show sticks, cd_host
set transparency, 0.30, cd_host
orient
zoom cd_host, 5
bg_color white

# Export 1920x1080 PNG
png HP_beta_CD_region_color.png, width=1920, height=1080, dpi=300, ray=1