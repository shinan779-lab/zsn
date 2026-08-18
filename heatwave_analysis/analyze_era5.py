#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERA5 4-6月 Tmax/Tmin 月平均异常与百分位分析 (1979-2026)
=================================================================
背景: 2026年6月欧洲严重热浪(法国44.3°C, 德国41.7°C等) + 夜间高温事件。
本脚本复现论文图件的核心计算:

  小时级 t2m  ->  逐日 Tmax/Tmin  ->  逐月平均  ->  2026年月平均异常(K)
                                              ->  2026年月平均百分位(%)

用法示例:
  # 演示模式(无需数据, 生成合成数据验证整条流程, 可先跑这个)
  python analyze_era5.py --demo --outdir output

  # 真实数据: --data 指向存放小时级 t2m 的 NetCDF 目录(可含多个文件)
  python analyze_era5.py --data /path/to/era5_hourly --outdir output

  # 若数据已经是"月平均"级别(每个文件为一个月), 加 --level monthly
  python analyze_era5.py --data /path/to/monthly --level monthly --outdir output

选项:
  --region global | europe   作图范围(默认 global=北半球; europe=欧洲放大)
  --clim 1979-2026 | 1979-2000  气候态参考期(默认 1979-2026, 与你原方法一致;
                                可额外用 1979-2000 做"旧气候态"对照)
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import xarray as xr
from shapely.geometry import shape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

MONTHS = [4, 5, 6]
MONTH_NAMES = {4: "April", 5: "May", 6: "June"}
VARS = ["tmax", "tmin"]
VAR_TITLES = {"tmax": "Tmax", "tmin": "Tmin"}

# 欧洲热浪关注区(Copernicus 西欧定义 11°W-15°E, 37°-55°N, 稍作扩展)
EU_BOX = dict(lon=slice(-15, 40), lat=slice(30, 72))
NH_BOX = dict(lon=slice(-180, 180), lat=slice(0, 90))


def _norm_time(ds):
    """新版 CDS API 下载的 ERA5 时间坐标名为 valid_time, 统一为 time."""
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    return ds


def _norm_lon(ds):
    """把 0-360 经度转到 -180-180."""
    if "lon" in ds.coords and float(ds.lon.max()) > 180:
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
        ds = ds.sortby("lon")
    return ds


def build_monthly(ds):
    """小时级(或月级)温度 -> 月平均 Tmax/Tmin, 维度 (time, lat, lon), 两个变量."""
    ds = _norm_time(ds)
    if "t2m" in ds.data_vars:
        var = "t2m"
    else:
        cand = [v for v in ("tasmax", "tasmin", "tmax", "tmin") if v in ds.data_vars]
        if not cand:
            raise KeyError("数据中未找到 t2m / tasmax / tasmin, 请检查变量名")
        var = cand[0]

    times = ds["time"].values.astype("datetime64[ns]").astype("int64")
    dt_h = np.median(np.diff(times)) / 3.6e12  # 小时

    if dt_h >= 20 * 24:  # 输入已是月平均
        tmax = ds[var].resample(time="1MS").mean("time")
        tmin = tmax.copy()
    else:                # 小时级: 先逐日极值, 再逐月平均
        tmax = ds[var].resample(time="1D").max("time").resample(time="1MS").max("time")
        tmin = ds[var].resample(time="1D").min("time").resample(time="1MS").min("time")

    tmax = tmax.rename("tmax")
    tmin = tmin.rename("tmin")
    out = _norm_lon(xr.merge([tmax, tmin]))
    out = out.assign_coords(year=("time", out.time.dt.year.values),
                            month=("time", out.time.dt.month.values))
    return out


def load_from_dir(data_dir):
    """读取目录下所有 nc 文件, 筛选 4-6 月, 计算月平均 Tmax/Tmin."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
    if not files:
        raise FileNotFoundError(f"{data_dir} 下没有 .nc 文件")
    ds = xr.open_mfdataset(files, combine="by_coords", compat="override")
    ds = ds.sel(time=ds.time.dt.month.isin(MONTHS))
    return build_monthly(ds)


def synthetic_demo():
    """合成 1979-2026 年 4-6 月月平均 Tmax/Tmin.
    2026 年加入西欧热浪信号(6月最强, 5月中等, 4月弱; Tmin 异常略弱于 Tmax),
    并含 0.04 K/yr 的升温趋势与逐年噪声, 用于验证整条分析流水线."""
    lat = np.arange(-90, 90.01, 2.5)
    lon = np.arange(-180, 180.01, 2.5)
    LON, LAT = np.meshgrid(lon, lat)
    years = np.arange(1979, 2027)
    rng = np.random.default_rng(42)
    ny, nm, nl, nlo = len(years), len(MONTHS), len(lat), len(lon)

    data = np.zeros((ny, nm, nl, nlo, 2))  # last dim: tmax, tmin
    for mi, m in enumerate(MONTHS):
        for yi, y in enumerate(years):
            trend = 0.04 * (y - 1979)
            noise = rng.normal(0, 0.9, (nl, nlo))
            blob_tmax = blob_tmin = 0.0
            if y == 2026:
                amp_tmax = {0: 1.2, 1: 3.6, 2: 6.0}[mi]
                amp_tmin = {0: 0.8, 1: 2.2, 2: 4.2}[mi]
                blob_tmax = amp_tmax * np.exp(-((LAT - 48) / 9) ** 2 - ((LON - 3) / 13) ** 2) \
                    + 0.5 * amp_tmax * np.exp(-((LAT - 52) / 8) ** 2 - ((LON - 16) / 12) ** 2)
                blob_tmin = amp_tmin * np.exp(-((LAT - 48) / 9) ** 2 - ((LON - 3) / 13) ** 2) \
                    + 0.5 * amp_tmin * np.exp(-((LAT - 52) / 8) ** 2 - ((LON - 16) / 12) ** 2)
            clim_tmax = 26 - 0.48 * (LAT - 10)
            data[yi, mi, :, :, 0] = clim_tmax + trend + noise + blob_tmax
            data[yi, mi, :, :, 1] = clim_tmax - 8.0 + trend + noise + blob_tmin

    return xr.Dataset(
        {"tmax": (("year", "month", "lat", "lon"), data[:, :, :, :, 0]),
         "tmin": (("year", "month", "lat", "lon"), data[:, :, :, :, 1])},
        coords={"year": years, "month": MONTHS, "lat": lat, "lon": lon},
    )


def percentile_map(da, target_year=2026):
    """逐格点计算 target_year 在全部年份中的百分位(0-100).
    da 维度为 (year, lat, lon)."""
    years = da.year.values
    idx = int(np.where(years == target_year)[0][0])
    v = da.values
    below = (v < v[idx]).sum(axis=0)
    equal = (v == v[idx]).sum(axis=0)
    return 100.0 * (below + 0.5 * equal) / len(years)


_WORLD_GEOMS = None


def load_world_borders():
    """加载内置国家边界 GeoJSON(放在本文件同目录 data/ 下), 用于绘制海岸线/国界.
    沙盒或离线环境无法访问 Natural Earth 服务器时, 用它替代 cartopy coastlines."""
    global _WORLD_GEOMS
    if _WORLD_GEOMS is not None:
        return _WORLD_GEOMS
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                        "countries.geo.json")
    if not os.path.exists(path):
        _WORLD_GEOMS = []
        return _WORLD_GEOMS
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    geoms = []
    for feat in gj["features"]:
        if feat.get("geometry"):
            geoms.append(shape(feat["geometry"]))
    _WORLD_GEOMS = geoms
    return geoms


def _draw(ax, lons, lats, field, cmap, vmin, vmax, pctl=None, title=""):
    ax.pcolormesh(lons, lats, field, cmap=cmap, vmin=vmin, vmax=vmax,
                  transform=ccrs.PlateCarree(), shading="auto")
    if pctl is not None:  # 90 百分位等值线
        ax.contour(lons, lats, pctl, levels=[90], colors="k", linewidths=0.8,
                   transform=ccrs.PlateCarree())
    geoms = load_world_borders()
    if geoms:
        ax.add_geometries(geoms, ccrs.PlateCarree(), facecolor="none",
                          edgecolor="0.35", linewidth=0.35)
    else:  # 没有本地边界数据时退回 cartopy 在线海岸线
        try:
            ax.coastlines(resolution="110m", linewidth=0.4)
        except Exception:
            pass
    ax.set_title(title, fontsize=12)
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.3,
                 color="gray", alpha=0.5)


def make_figures(ds, clim_years, region, outdir, tag):
    """ds: (year, month, lat, lon); 返回异常与百分位 dict, 并画两张 3x2 图."""
    ds = ds.sel(year=clim_years)
    box = EU_BOX if region == "europe" else NH_BOX
    lats = ds.lat.values
    lons = ds.lon.values

    anom = {}
    pctl = {}
    vmax = 0.0
    for var in VARS:
        anom[var] = {}
        pctl[var] = {}
        for m in MONTHS:
            da_m = ds[var].sel(month=m)
            a = da_m.sel(year=2026) - da_m.mean("year")
            anom[var][m] = a
            pctl[var][m] = percentile_map(da_m)
            vmax = max(vmax, float(np.nanmax(np.abs(a.values))))

    vmax = float(np.ceil(vmax * 2) / 2)  # 向上取整到 0.5

    # ---- 图1: 异常 (K) ----
    fig, axes = plt.subplots(3, 2, figsize=(13, 16),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    mappable = None
    for r, m in enumerate(MONTHS):
        for c, var in enumerate(VARS):
            ax = axes[r, c]
            a = anom[var][m].sel(lat=box["lat"], lon=box["lon"])
            p = xr.DataArray(pctl[var][m], coords={"lat": lats, "lon": lons}).sel(
                lat=box["lat"], lon=box["lon"])
            _draw(ax, a.lon.values, a.lat.values, a.values, "RdBu_r", -vmax, vmax,
                  pctl=p.values, title=f"{MONTH_NAMES[m]} {VAR_TITLES[var]} anomaly (K)")
            if mappable is None:
                mappable = ax.collections[-1]
    fig.colorbar(mappable, ax=axes.ravel().tolist(), orientation="horizontal",
                 pad=0.04, shrink=0.6, label="K", extend="both")
    fig.suptitle(f"2026 {MONTH_NAMES[4]}-{MONTH_NAMES[6]} Tmax/Tmin monthly anomalies "
                 f"(clim {clim_years[0]}-{clim_years[-1]})\n"
                 "black contour = 90th percentile of the reference period", fontsize=13)
    fig.savefig(os.path.join(outdir, f"anomaly_{tag}_{region}.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- 图2: 百分位 (%) ----
    fig, axes = plt.subplots(3, 2, figsize=(13, 16),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    mappable = None
    for r, m in enumerate(MONTHS):
        for c, var in enumerate(VARS):
            ax = axes[r, c]
            p = xr.DataArray(pctl[var][m], coords={"lat": lats, "lon": lons}).sel(
                lat=box["lat"], lon=box["lon"])
            _draw(ax, p.lon.values, p.lat.values, p.values, "YlOrRd", 0, 100,
                  title=f"{MONTH_NAMES[m]} {VAR_TITLES[var]} percentile (%)")
            if mappable is None:
                mappable = ax.collections[-1]
    fig.colorbar(mappable, ax=axes.ravel().tolist(), orientation="horizontal",
                 pad=0.04, shrink=0.6, label="percentile (%)")
    fig.suptitle(f"2026 {MONTH_NAMES[4]}-{MONTH_NAMES[6]} Tmax/Tmin monthly percentile "
                 f"within {clim_years[0]}-{clim_years[-1]}", fontsize=13)
    fig.savefig(os.path.join(outdir, f"percentile_{tag}_{region}.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return anom, pctl


def report_stats(ds, anom, pctl, outdir, tag, region):
    """输出关注区统计: 面积平均异常、>90百分位格点占比、最强格点."""
    lats = ds.lat.values
    lons = ds.lon.values
    rows = []
    for var in VARS:
        for m in MONTHS:
            a_full = anom[var][m]  # 全图异常
            p_full = xr.DataArray(pctl[var][m], coords={"lat": lats, "lon": lons})
            a = a_full.sel(lat=EU_BOX["lat"], lon=EU_BOX["lon"])
            p = p_full.sel(lat=EU_BOX["lat"], lon=EU_BOX["lon"])
            w = np.cos(np.deg2rad(a.lat.values))[:, None]
            amean = float(np.average(a.values, weights=np.broadcast_to(w, a.values.shape)))
            frac90 = float((p.values > 90).mean()) * 100
            amax = a.values
            flat = int(np.nanargmax(amax))
            iy, ix = np.unravel_index(flat, amax.shape)
            rows.append([f"{MONTH_NAMES[m]} {VAR_TITLES[var]}", amean, frac90,
                         float(a.lat.values[iy]), float(a.lon.values[ix]),
                         float(amax[iy, ix]), float(p.values[iy, ix])])
    with open(os.path.join(outdir, f"stats_{tag}_{region}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["month_var", "EU_box_mean_anom_K", "EU_frac>90pct_%",
                    "max_cell_lat", "max_cell_lon", "max_anom_K", "max_pctl"])
        w.writerows(rows)
    print("\n=== 统计结果(欧洲关注区 15W-40E, 30-72N) ===")
    print(f"{'month_var':<18}{'区域均异常(K)':>14}{'>90pct占比(%)':>14}{'最强格点':>24}")
    for r in rows:
        print(f"{r[0]:<18}{r[1]:>14.2f}{r[2]:>14.1f}"
              f"  ({r[3]:.0f}N, {r[4]:.0f}E): {r[5]:.2f}K (pctl {r[6]:.0f})")


def main():
    ap = argparse.ArgumentParser(description="ERA5 4-6月 Tmax/Tmin 月平均异常与百分位分析")
    ap.add_argument("--data", help="小时级(或月级) ERA5 NetCDF 目录")
    ap.add_argument("--demo", action="store_true", help="用合成数据演示")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--region", choices=["global", "europe"], default="global")
    ap.add_argument("--clim", default="1979-2026", help="气候态参考期, 如 1979-2026 / 1979-2000")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    y0, y1 = map(int, args.clim.split("-"))
    clim_years = np.arange(y0, y1 + 1)

    if args.demo:
        ds = synthetic_demo()
        tag = "demo"
    else:
        if not args.data:
            ap.error("请提供 --data 或使用 --demo")
        ds = load_from_dir(args.data)
        tag = "era5"

    anom, pctl = make_figures(ds, clim_years, args.region, args.outdir, tag)
    report_stats(ds, anom, pctl, args.outdir, tag, args.region)
    print(f"\n输出文件已保存到: {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
