# CD Region-colour attention map
# grey=Backbone, orange=Cavity, green=Primary rim, blue=Secondary rim

load results/pymol/gamma_CD.pdb, cd_host

# Define region colours
set_color col_backbone, [0.533, 0.533, 0.533]
set_color col_cavity, [0.878, 0.482, 0.224]
set_color col_primary, [0.298, 0.686, 0.314]
set_color col_secondary, [0.357, 0.608, 0.835]

color col_backbone, cd_host

color col_cavity, (cd_host and index 3+8+15+22+29+36+43+50+57+63+67+71+75+79+83+87)
color col_primary, (cd_host and index 1+2+9+10+16+17+23+24+30+31+37+38+44+45+51+52)
color col_secondary, (cd_host and index 58+59+60+61+62+64+65+66+68+69+70+72+73+74+76+77+78+80+81+82+84+85+86+88)

hide everything
show surface, cd_host
show sticks, cd_host
set transparency, 0.30, cd_host
orient
zoom cd_host, 5
bg_color white