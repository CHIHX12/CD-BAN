# CD-BAN Attention Visualization
# Naproxen inside HP-beta-CD
# Colour = normalised BANLayer attention (blue=low, red=high)

# ── 載入結構 ────────────────────────
load /home/nibiohnproj9/cycheng/cd_gnn_training/CD_BAN/results/pymol/HP_beta_CD.pdb, cd_host
load /home/nibiohnproj9/cycheng/cd_gnn_training/CD_BAN/results/pymol/naproxen.pdb, guest

# ── 初始化 B-factor = 0 ─────────────
alter cd_host, b=0.0

# ── 注意力分數 → B-factor ────────────
alter (cd_host and index 1), b=1.000000
alter (cd_host and index 2), b=0.800742
alter (cd_host and index 3), b=0.925748
alter (cd_host and index 4), b=0.868877
alter (cd_host and index 5), b=0.871776
alter (cd_host and index 6), b=0.923394
alter (cd_host and index 7), b=0.895654
alter (cd_host and index 8), b=0.932281
alter (cd_host and index 9), b=0.926494
alter (cd_host and index 10), b=0.926865
alter (cd_host and index 11), b=0.904954
alter (cd_host and index 12), b=0.826504
alter (cd_host and index 13), b=0.922417
alter (cd_host and index 14), b=0.834945
alter (cd_host and index 15), b=0.915461
alter (cd_host and index 16), b=0.926494
alter (cd_host and index 17), b=0.926865
alter (cd_host and index 18), b=0.904954
alter (cd_host and index 19), b=0.826504
alter (cd_host and index 20), b=0.922417
alter (cd_host and index 21), b=0.834945
alter (cd_host and index 22), b=0.915461
alter (cd_host and index 23), b=0.926494
alter (cd_host and index 24), b=0.926865
alter (cd_host and index 25), b=0.904954
alter (cd_host and index 26), b=0.826504
alter (cd_host and index 27), b=0.922417
alter (cd_host and index 28), b=0.834945
alter (cd_host and index 29), b=0.915461
alter (cd_host and index 30), b=0.926494
alter (cd_host and index 31), b=0.926865
alter (cd_host and index 32), b=0.904954
alter (cd_host and index 33), b=0.826504
alter (cd_host and index 34), b=0.922417
alter (cd_host and index 35), b=0.834945
alter (cd_host and index 36), b=0.915461
alter (cd_host and index 37), b=0.926494
alter (cd_host and index 38), b=0.926865
alter (cd_host and index 39), b=0.904954
alter (cd_host and index 40), b=0.826504
alter (cd_host and index 41), b=0.922417
alter (cd_host and index 42), b=0.834945
alter (cd_host and index 43), b=0.915461
alter (cd_host and index 44), b=0.926494
alter (cd_host and index 45), b=0.926865
alter (cd_host and index 46), b=0.904954
alter (cd_host and index 47), b=0.826504
alter (cd_host and index 48), b=0.922417
alter (cd_host and index 49), b=0.834945
alter (cd_host and index 50), b=0.915461
alter (cd_host and index 51), b=0.926494
alter (cd_host and index 52), b=0.926865
alter (cd_host and index 53), b=0.904954
alter (cd_host and index 54), b=0.826504
alter (cd_host and index 55), b=0.922417
alter (cd_host and index 56), b=0.834945
alter (cd_host and index 57), b=0.915461
alter (cd_host and index 58), b=0.932281
alter (cd_host and index 59), b=0.895654
alter (cd_host and index 60), b=0.923394
alter (cd_host and index 61), b=0.871776
alter (cd_host and index 62), b=0.868877
alter (cd_host and index 63), b=0.800742
alter (cd_host and index 64), b=1.000000
alter (cd_host and index 65), b=0.925748
alter (cd_host and index 66), b=0.932281
alter (cd_host and index 67), b=0.895654
alter (cd_host and index 68), b=0.923394
alter (cd_host and index 69), b=0.871776
alter (cd_host and index 70), b=0.868877
alter (cd_host and index 71), b=0.800742
alter (cd_host and index 72), b=1.000000
alter (cd_host and index 73), b=0.925748
alter (cd_host and index 74), b=0.932281
alter (cd_host and index 75), b=0.895654
alter (cd_host and index 76), b=0.923394
alter (cd_host and index 77), b=0.871776
alter (cd_host and index 78), b=0.868877
alter (cd_host and index 79), b=0.800742
alter (cd_host and index 80), b=1.000000
alter (cd_host and index 81), b=0.925748
alter (cd_host and index 82), b=0.932281
alter (cd_host and index 83), b=0.895654
alter (cd_host and index 84), b=0.923394
alter (cd_host and index 85), b=0.871776
alter (cd_host and index 86), b=0.868877
alter (cd_host and index 87), b=0.800742
alter (cd_host and index 88), b=1.000000
alter (cd_host and index 89), b=0.925748
alter (cd_host and index 90), b=0.932281
alter (cd_host and index 91), b=0.895654
alter (cd_host and index 92), b=0.923394
alter (cd_host and index 93), b=0.871776
alter (cd_host and index 94), b=0.868877
alter (cd_host and index 95), b=0.800742
alter (cd_host and index 96), b=1.000000
alter (cd_host and index 97), b=0.925748
alter (cd_host and index 98), b=0.932281
alter (cd_host and index 99), b=0.895654
alter (cd_host and index 100), b=0.923394
alter (cd_host and index 101), b=0.871776
alter (cd_host and index 102), b=0.868877
alter (cd_host and index 103), b=0.800742
alter (cd_host and index 104), b=1.000000
alter (cd_host and index 105), b=0.925748

# ── 上色 ────────────────────────────
# CD 依注意力上色（藍=低注意力，紅=高注意力）
spectrum b, blue_white_red, cd_host, minimum=0, maximum=1

# ── 顯示風格 ─────────────────────────
hide everything
show surface, cd_host
show sticks, cd_host
set transparency, 0.35, cd_host

# 配體（藥物）顯示
show sticks, guest
color yellow, guest
set stick_radius, 0.15, guest

# ── 相機設定 ─────────────────────────
orient
zoom cd_host, 5
set bg_color, white

# ── 顏色圖例說明 ─────────────────────
# 藍色 = CD 原子注意力低（模型較不關注）
# 紅色 = CD 原子注意力高（模型認為此區域影響結合）
# 黃色 = Naproxen（藥物配體）

# 存圖（可選）：
# png results/pymol/HP_beta_CD_naproxen.png, width=1200, height=900, dpi=300