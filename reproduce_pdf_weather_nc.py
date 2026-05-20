"""Reproduce the PDF weather-composite workflow with yearly NetCDF weather inputs.

This script mirrors the logic in Fire_Masks_n_Weather_composites.ipynb:
- For each fire event, use the 28 days before fire start (inclusive).
- Aggregate daily weather rasters with mean or sum.
- Export one GeoTIFF per fire and predictor, clipped/resampled to the S2 tile extent.

Inputs:
1) Fire shapefile with fields: FireNo, StartDate (YYYY-MM-DD)
2) Yearly NetCDF files per predictor (one file can contain many days)
3) Tile extent definition (xmin ymin xmax ymax in target CRS)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
import xarray as xr


@dataclass
class TileConfig:
    tile: str
    extent: Tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    out_res: float = 20.0


TILE_EXTENTS: Dict[str, TileConfig] = {
    "T56HKJ": TileConfig("T56HKJ", (199980.0, 6290200.0, 309780.0, 6400000.0)),
    "T56HKH": TileConfig("T56HKH", (199980.0, 6190240.0, 309780.0, 6300040.0)),
    "T56HKG": TileConfig("T56HKG", (199980.0, 6090220.0, 309780.0, 6200020.0)),
}


def find_yearly_nc_files(root: Path, predictor: str, tile: str) -> List[Path]:
    """Find yearly NetCDF files for one predictor/tile."""
    p = root / f"{predictor}_{tile}"
    return sorted(p.rglob("*.nc"))


def _detect_var(ds: xr.Dataset, preferred: Optional[str]) -> str:
    if preferred and preferred in ds.data_vars:
        return preferred
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise ValueError(f"Unable to detect variable; dataset has vars {list(ds.data_vars)}")


def _detect_time_name(ds: xr.Dataset) -> str:
    for key in ["time", "Time", "date", "datetime"]:
        if key in ds.coords:
            return key
    raise ValueError("No time coordinate found.")


def aggregate_fire_window(
    nc_files: Iterable[Path],
    start_date: datetime,
    days: int,
    agg: str,
    var_name: Optional[str] = None,
) -> xr.DataArray:
    """Load all yearly NCs, filter fire window, aggregate across time."""
    ds = xr.open_mfdataset([str(p) for p in nc_files], combine="by_coords")
    var = _detect_var(ds, var_name)
    t = _detect_time_name(ds)

    t0 = np.datetime64((start_date - timedelta(days=days)).strftime("%Y-%m-%d"))
    t1 = np.datetime64(start_date.strftime("%Y-%m-%d"))
    window = ds[var].sel({t: slice(t0, t1)})
    if window.sizes.get(t, 0) == 0:
        raise ValueError(f"No weather slices found between {t0} and {t1}")

    if agg == "avg":
        out = window.mean(dim=t, skipna=True)
    elif agg == "sum":
        out = window.sum(dim=t, skipna=True)
    else:
        raise ValueError("agg must be 'avg' or 'sum'")
    return out


def _xy_names(da: xr.DataArray) -> Tuple[str, str]:
    x_cands = [c for c in ["x", "lon", "longitude"] if c in da.coords]
    y_cands = [c for c in ["y", "lat", "latitude"] if c in da.coords]
    if not x_cands or not y_cands:
        raise ValueError(f"Unable to infer XY coords from {list(da.coords)}")
    return x_cands[0], y_cands[0]


def write_geotiff_from_dataarray(
    da: xr.DataArray,
    out_tif: Path,
    tile_cfg: TileConfig,
    dst_crs: str,
    nodata: float = -32768.0,
) -> None:
    """Export 2D DataArray as GeoTIFF on fixed tile extent/grid."""
    x_name, y_name = _xy_names(da)
    arr = da.values.astype(np.float32)
    arr = np.where(np.isnan(arr), nodata, arr)

    xmin, ymin, xmax, ymax = tile_cfg.extent
    width = int(round((xmax - xmin) / tile_cfg.out_res))
    height = int(round((ymax - ymin) / tile_cfg.out_res))
    dst_transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    src_x = da.coords[x_name].values
    src_y = da.coords[y_name].values
    src_transform = from_bounds(float(src_x.min()), float(src_y.min()), float(src_x.max()), float(src_y.max()), arr.shape[1], arr.shape[0])

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": nodata,
        "width": width,
        "height": height,
        "count": 1,
        "crs": dst_crs,
        "transform": dst_transform,
        "compress": "deflate",
    }
    dst = np.full((height, width), nodata, dtype=np.float32)

    rasterio.warp.reproject(
        source=arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=dst_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=nodata,
        dst_nodata=nodata,
        resampling=Resampling.nearest,
    )

    with rasterio.open(out_tif, "w", **profile) as out:
        out.write(dst, 1)


def run(args: argparse.Namespace) -> None:
    tile_cfg = TILE_EXTENTS[args.tile]
    fires = gpd.read_file(args.fire_shp)
    fires["StartDate"] = pd.to_datetime(fires["StartDate"]).dt.strftime("%Y-%m-%d")

    for predictor in args.predictors:
        nc_files = find_yearly_nc_files(Path(args.weather_root), predictor, args.tile)
        if not nc_files:
            print(f"[WARN] No NC files found for {predictor}, skip")
            continue

        for _, row in fires.iterrows():
            fire_no = str(row["FireNo"])
            start = datetime.strptime(row["StartDate"], "%Y-%m-%d")
            print(f"[{predictor}] fire={fire_no}, start={row['StartDate']}")

            agg_da = aggregate_fire_window(
                nc_files=nc_files,
                start_date=start,
                days=args.window_days,
                agg=args.mode,
                var_name=args.nc_var,
            )
            out_tif = Path(args.output_root) / f"{args.tile}_Weather" / predictor / f"{predictor}_{fire_no}_{row['StartDate']}_{args.mode}.tif"
            write_geotiff_from_dataarray(
                da=agg_da,
                out_tif=out_tif,
                tile_cfg=tile_cfg,
                dst_crs=args.crs,
                nodata=args.nodata,
            )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reproduce weather composite workflow with yearly NC files")
    p.add_argument("--fire-shp", required=True, help="FireHistory shapefile path (with FireNo, StartDate)")
    p.add_argument("--weather-root", required=True, help="Root folder containing <Predictor>_<Tile>/*.nc")
    p.add_argument("--output-root", required=True)
    p.add_argument("--tile", required=True, choices=sorted(TILE_EXTENTS.keys()))
    p.add_argument("--predictors", nargs="+", required=True)
    p.add_argument("--mode", default="avg", choices=["avg", "sum"])
    p.add_argument("--window-days", type=int, default=28)
    p.add_argument("--nc-var", default=None, help="NC variable name; auto-detect if omitted")
    p.add_argument("--crs", default="EPSG:32756", help="Target CRS for output GeoTIFF")
    p.add_argument("--nodata", type=float, default=-32768.0)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
