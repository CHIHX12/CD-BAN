# CD-BAN attention on HP-beta-CD (RDKit ETKDGv3+MMFF structure)
# Color: blue=low, white=mid, red=high attention

load HP_beta_CD_rdkit.pdb, cd_host

alter cd_host, b=0.0

# Map attention by atom index
alter (cd_host and index 1), b=0.542154
alter (cd_host and index 2), b=0.542154
alter (cd_host and index 3), b=0.542154
alter (cd_host and index 4), b=0.542154
alter (cd_host and index 5), b=0.542154
alter (cd_host and index 6), b=0.542154
alter (cd_host and index 7), b=0.628931
alter (cd_host and index 8), b=0.747551
alter (cd_host and index 9), b=0.531949
alter (cd_host and index 10), b=0.648823
alter (cd_host and index 11), b=0.483863
alter (cd_host and index 12), b=0.121884
alter (cd_host and index 13), b=0.690487
alter (cd_host and index 14), b=0.138190
alter (cd_host and index 15), b=0.664303
alter (cd_host and index 16), b=0.531949
alter (cd_host and index 17), b=0.648823
alter (cd_host and index 18), b=0.483863
alter (cd_host and index 19), b=0.121884
alter (cd_host and index 20), b=0.690487
alter (cd_host and index 21), b=0.138190
alter (cd_host and index 22), b=0.664303
alter (cd_host and index 23), b=0.531949
alter (cd_host and index 24), b=0.648823
alter (cd_host and index 25), b=0.483863
alter (cd_host and index 26), b=0.121884
alter (cd_host and index 27), b=0.690487
alter (cd_host and index 28), b=0.138190
alter (cd_host and index 29), b=0.664303
alter (cd_host and index 30), b=0.531949
alter (cd_host and index 31), b=0.648823
alter (cd_host and index 32), b=0.483863
alter (cd_host and index 33), b=0.121884
alter (cd_host and index 34), b=0.690487
alter (cd_host and index 35), b=0.138190
alter (cd_host and index 36), b=0.664303
alter (cd_host and index 37), b=0.531949
alter (cd_host and index 38), b=0.648823
alter (cd_host and index 39), b=0.483863
alter (cd_host and index 40), b=0.121884
alter (cd_host and index 41), b=0.690487
alter (cd_host and index 42), b=0.138190
alter (cd_host and index 43), b=0.664303
alter (cd_host and index 44), b=0.531949
alter (cd_host and index 45), b=0.648823
alter (cd_host and index 46), b=0.483863
alter (cd_host and index 47), b=0.121884
alter (cd_host and index 48), b=0.690487
alter (cd_host and index 49), b=0.138190
alter (cd_host and index 50), b=0.664303
alter (cd_host and index 51), b=0.531949
alter (cd_host and index 52), b=0.648823
alter (cd_host and index 53), b=0.483863
alter (cd_host and index 54), b=0.121884
alter (cd_host and index 55), b=0.690487
alter (cd_host and index 56), b=0.138190
alter (cd_host and index 57), b=0.664303
alter (cd_host and index 58), b=0.747551
alter (cd_host and index 59), b=0.628931
alter (cd_host and index 60), b=0.542154
alter (cd_host and index 61), b=0.542154
alter (cd_host and index 62), b=0.542154
alter (cd_host and index 63), b=0.542154
alter (cd_host and index 64), b=0.542154
alter (cd_host and index 65), b=0.542154
alter (cd_host and index 66), b=0.747551
alter (cd_host and index 67), b=0.628931
alter (cd_host and index 68), b=0.542154
alter (cd_host and index 69), b=0.542154
alter (cd_host and index 70), b=0.542154
alter (cd_host and index 71), b=0.542154
alter (cd_host and index 72), b=0.542154
alter (cd_host and index 73), b=0.542154
alter (cd_host and index 74), b=0.747551
alter (cd_host and index 75), b=0.628931
alter (cd_host and index 76), b=0.542154
alter (cd_host and index 77), b=0.542154
alter (cd_host and index 78), b=0.542154
alter (cd_host and index 79), b=0.542154
alter (cd_host and index 80), b=0.542154
alter (cd_host and index 81), b=0.542154
alter (cd_host and index 82), b=0.747551
alter (cd_host and index 83), b=0.628931
alter (cd_host and index 84), b=0.542154
alter (cd_host and index 85), b=0.542154
alter (cd_host and index 86), b=0.542154
alter (cd_host and index 87), b=0.542154
alter (cd_host and index 88), b=0.542154
alter (cd_host and index 89), b=0.542154
alter (cd_host and index 90), b=0.747551
alter (cd_host and index 91), b=0.628931
alter (cd_host and index 92), b=0.542154
alter (cd_host and index 93), b=0.542154
alter (cd_host and index 94), b=0.542154
alter (cd_host and index 95), b=0.542154
alter (cd_host and index 96), b=0.542154
alter (cd_host and index 97), b=0.542154
alter (cd_host and index 98), b=0.747551
alter (cd_host and index 99), b=0.628931
alter (cd_host and index 100), b=0.542154
alter (cd_host and index 101), b=0.542154
alter (cd_host and index 102), b=0.542154
alter (cd_host and index 103), b=0.542154
alter (cd_host and index 104), b=0.542154
alter (cd_host and index 105), b=0.542154

spectrum b, blue_white_red, cd_host, minimum=0, maximum=1
hide everything
show sticks, cd_host
show surface, cd_host
set transparency, 0.30, cd_host
set stick_radius, 0.15
orient cd_host
zoom cd_host, 3
set bg_color, white
ray 1920, 1080
png HP_beta_CD_crystal_render.png, dpi=300

# blue=low(C2,C3)  white=mid  red=high(O5,O2,O3,Ob)