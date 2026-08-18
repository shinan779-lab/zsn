#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量分析 rankScore 地图图片: 提取色标、统计颜色占比、生成空间分布地图."""
import os
import numpy as np
from PIL import Image


def classify(img_path):
    """分类像素: 返回 (green, magenta, deep_purple, white, land_yellow) 掩码."""
    a = np.asarray(Image.open(img_path).convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    white = (r > 235) & (g > 235) & (b > 235)
    green = (g > 90) & (g > r + 20) & (g > b + 20)
    magenta = (r > 140) & (b > 80) & (g < r - 40)
    deep_purple = (b > 100) & (b > g + 40) & (r < 200)
    land_yellow = (np.abs(r - 255) < 15) & (np.abs(g - 230) < 15) & (np.abs(b - 153) < 15)
    return a, white, green, magenta, deep_purple, land_yellow


def find_colorbar(a, y_range, x_range, label):
    """在给定区域找最长的横向彩色渐变条(绿段+品红段), 返回其位置与两端颜色."""
    y0, y1 = y_range
    x0, x1 = x_range
    best = None
    for y in range(y0, y1):
        br = a[y, x0:x1, 0].astype(int)
        bg = a[y, x0:x1, 1].astype(int)
        bb = a[y, x0:x1, 2].astype(int)
        colored = (np.abs(br - bg) > 25) | (np.abs(bg - bb) > 25)
        colored = colored & ~((np.abs(br - 255) < 15) & (np.abs(bg - 230) < 15) & (np.abs(bb - 153) < 15))
        run_start, run_len, best_run = -1, 0, (0, 0, 0)
        in_run = False
        for x in range(len(colored)):
            if colored[x]:
                if not in_run:
                    in_run = True
                    run_start = x
                run_len = x - run_start + 1
            else:
                if in_run and run_len > best_run[0]:
                    best_run = (run_len, run_start, x - 1)
                in_run = False
                run_len = 0
        if in_run and run_len > best_run[0]:
            best_run = (run_len, run_start, len(colored) - 1)
        if best_run[0] > 80 and (best is None or best_run[0] > best[0]):
            best = (best_run[0], y, x0 + best_run[1], x0 + best_run[2])
    if not best:
        return None
    ln, y, xa, xb = best
    seg = a[y, xa:xb]
    # 找绿-品红分界(绿色段和品红段的分界点 = g 最小的位置附近)
    g_ = seg[:, 1].astype(int)
    r_ = seg[:, 0].astype(int)
    # 绿段: g>r; 品红段: r>g
    split = None
    for x in range(len(seg)):
        if g_[x] < r_[x]:
            split = x
            break
    return {"y": y, "xa": xa, "xb": xb, "width": ln, "split": split,
            "left_color": seg[0].tolist(),
            "mid_color": seg[len(seg) // 2].tolist(),
            "right_color": seg[-1].tolist(),
            "split_color": seg[split].tolist() if split else None}


def spatial_map(name, a, white, green, magenta, deep_purple, land,
                y0, y1, x0, x1, rows=16, cols=40, title=""):
    """打印空间分布字符图."""
    print(f"\n=== {name} {title} (y{y0}-{y1}, x{x0}-{x1}) ===")
    for i in range(rows):
        line = ""
        for j in range(cols):
            yy0 = y0 + (y1 - y0) * i // rows
            yy1 = y0 + (y1 - y0) * (i + 1) // rows
            xx0 = x0 + (x1 - x0) * j // cols
            xx1 = x0 + (x1 - x0) * (j + 1) // cols
            pf = deep_purple[yy0:yy1, xx0:xx1].mean()
            mf = magenta[yy0:yy1, xx0:xx1].mean()
            gf = green[yy0:yy1, xx0:xx1].mean()
            wf = white[yy0:yy1, xx0:xx1].mean()
            lf = land[yy0:yy1, xx0:xx1].mean()
            if pf > 0.12:
                line += "P"
            elif mf > 0.12:
                line += "M"
            elif gf > 0.12:
                line += "G"
            elif gf > 0.04 or mf > 0.04 or pf > 0.04:
                line += "."
            elif lf > 0.3:
                line += "="
            elif wf > 0.6:
                line += " "
            else:
                line += "·"
        print(line)


def analyze(path, label):
    print(f"\n{'#'*70}\n# {label}\n# 文件: {os.path.basename(path)}\n{'#'*70}")
    a, white, green, magenta, deep_purple, land = classify(path)
    h, w, _ = a.shape
    print(f"尺寸: {w}x{h}")
    total_px = h * w
    for nm, mask in [("白色", white), ("绿色", green), ("品红", magenta),
                     ("深紫", deep_purple), ("米黄", land)]:
        print(f"  {nm}: {mask.mean()*100:6.2f}%")

    # 找色标条 (在底部区域或右部)
    cb = find_colorbar(a, (int(h*0.85), h), (0, w), label)
    if cb:
        print(f"色标条: y={cb['y']}, x={cb['xa']}-{cb['xb']} (宽{cb['width']}px)")
        print(f"  左端(绿): RGB{cb['left_color']}  中段: RGB{cb['mid_color']}  右端(品红): RGB{cb['right_color']}")
        if cb['split']:
            print(f"  绿-品红分界在 x={cb['xa']+cb['split']} (距左 {cb['split']}px)")
    return a, white, green, magenta, deep_purple, land


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    files = [
        ("TXmean_rankScore_2026_04_land.png", "Tmax 2026-04"),
        ("TXmean_rankScore_2026_05_land.png", "Tmax 2026-05"),
        ("TXmean_rankScore_2026_06_land.png", "Tmax 2026-06"),
        ("TNmean_rankScore_2026_04_land.png", "Tmin 2026-04"),
        ("TNmean_rankScore_2026_05_land.png", "Tmin 2026-05"),
        ("TNmean_rankScore_2026_06_land.png", "Tmin 2026-06"),
    ]
    for fn, label in files:
        p = os.path.join(data_dir, fn)
        analyze(p, label)
