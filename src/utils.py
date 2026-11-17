"""Shared helpers: configuration, raster input/output, grids, polygons and metrics.

All rasters are in EPSG:32645 (UTM zone 45N). A raster is handled as a NumPy array plus a
small georeference dict ``g`` with the upper-left corner (x0, y0), the pixel size (sx, sy)
and the nodata value. Rasterio is used when it is installed. A minimal GeoTIFF reader in
``_tiff_fallback.py`` is used otherwise, so the analysis also runs where rasterio is missing.
"""
import json
import os

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------- configuration
def load_config(path=None):
    path = path or os.path.join(ROOT, "config.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = ROOT
    return cfg


def p(cfg, *keys):
    """Absolute path of an entry under cfg['paths'] (nested keys allowed)."""
    v = cfg["paths"]
    for k in keys:
        v = v[k]
    return v if os.path.isabs(v) else os.path.join(ROOT, v)


def rp(path):
    """Absolute path for a path written relative to the repository root."""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def out(cfg, name):
    d = p(cfg, "output_dir")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


# ----------------------------------------------------------------------------- raster IO
def read_raster(path):
    """Return (array, g). Multi-band rasters come back as (bands, rows, cols)."""
    try:
        import rasterio
    except ImportError:
        from _tiff_fallback import read_any
        return read_any(path)
    with rasterio.open(path) as ds:
        a = ds.read()
        t = ds.transform
        g = dict(x0=t.c, y0=t.f, sx=t.a, sy=-t.e, nodata=ds.nodata)
    return (a[0] if a.shape[0] == 1 else a), g


def write_raster(path, a, g, nodata=None, dtype=None):
    """Write a single-band GeoTIFF (EPSG:32645). Falls back to .npy without rasterio."""
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        np.save(os.path.splitext(path)[0] + ".npy", a)
        return
    a = a.astype(dtype or a.dtype)
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1,
                       dtype=a.dtype, crs="EPSG:32645", nodata=nodata, compress="deflate",
                       transform=from_origin(g["x0"], g["y0"], g["sx"], g["sy"])) as ds:
        ds.write(a, 1)


# ----------------------------------------------------------------------------- satellite grid
def load_scene(path, grid_g=None, grid_shape=None):
    """PlanetScope 8-band scene as float32 (bands, H, W). If a reference grid is given, a
    clipped scene is pasted into it at its pixel offset (missing pixels stay 0)."""
    a, g = read_raster(path)
    a = a.astype(np.float32)
    if grid_g is None:
        return a, g
    H, W = grid_shape
    full = np.zeros((a.shape[0], H, W), np.float32)
    c0 = int(round((g["x0"] - grid_g["x0"]) / grid_g["sx"]))
    r0 = int(round((grid_g["y0"] - g["y0"]) / grid_g["sy"]))
    hh, ww = min(a.shape[1], H - r0), min(a.shape[2], W - c0)
    full[:, r0:r0 + hh, c0:c0 + ww] = a[:, :hh, :ww]
    return full, grid_g


# ----------------------------------------------------------------------------- geometry
def to_utm45(lon, lat):
    """WGS84 longitude/latitude to UTM zone 45N (EPSG:32645), standard series formulas."""
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    k0 = 0.9996
    lon0 = np.radians(87)
    lat = np.radians(lat)
    lon = np.radians(lon)
    ep2 = e2 / (1 - e2)
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    T = np.tan(lat) ** 2
    C = ep2 * np.cos(lat) ** 2
    A = np.cos(lat) * (lon - lon0)
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * lat)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * lat)
             - (35 * e2 ** 3 / 3072) * np.sin(6 * lat))
    x = k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000
    y = k0 * (M + N * np.tan(lat) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                                    + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720))
    return x, y


def load_polygons(path):
    """GeoJSON polygons as (properties, [(is_outer, Nx2 UTM array), ...]). Lon/lat input is projected."""
    g = json.load(open(path, encoding="utf-8"))
    crs = json.dumps(g.get("crs", {}))
    geographic = ("4326" in crs) or ("CRS84" in crs) or not g.get("crs")
    feats = []
    for ft in g["features"]:
        gm = ft["geometry"]
        polys = gm["coordinates"] if gm["type"] == "MultiPolygon" else [gm["coordinates"]]
        rings = []
        for poly in polys:
            for i, r in enumerate(poly):
                r = np.array(r)[:, :2]
                if geographic and abs(r[0, 0]) <= 180:
                    r = np.column_stack(to_utm45(r[:, 0], r[:, 1]))
                rings.append((i == 0, r))
        feats.append((ft["properties"], rings))
    return feats


def rasterize(rings, g, shape):
    from PIL import Image, ImageDraw
    H, W = shape
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for outer, r in rings:
        d.polygon([((x - g["x0"]) / g["sx"], (g["y0"] - y) / g["sy"]) for x, y in r], fill=1 if outer else 0)
    return np.array(im).astype(bool)


# ----------------------------------------------------------------------------- spectral helpers
EPS = 1e-6


def nd(a, b):
    return (a - b) / (a + b + EPS)


def ndvi(x):      # x: (8, H, W) reflectance; SuperDove band 6 = red, band 8 = NIR
    return nd(x[7], x[5])


def ndre(x):      # band 7 = red edge
    return nd(x[7], x[6])


def shift_bands(img, dy, dx):
    from scipy import ndimage
    if dy == 0 and dx == 0:
        return img
    return np.stack([ndimage.shift(b, (dy, dx), order=1, mode="nearest") for b in img])


def best_shift(ndvi_img, fracs, fields, candidates):
    """Co-registration shift (in 3 m pixels) that maximises the mean correlation between NDVI and
    the UAV canopy fraction over the given (training) fields only."""
    from scipy import ndimage
    best = None
    for dy, dx in candidates:
        s = ndimage.shift(ndvi_img, (dy, dx), order=1, mode="nearest")
        r = np.mean([np.corrcoef(s[np.isfinite(fracs[k])], fracs[k][np.isfinite(fracs[k])])[0, 1] for k in fields])
        if best is None or r > best[0]:
            best = (r, dy, dx)
    return best[1], best[2]


def best_threshold(prob, y, grid):
    from sklearn.metrics import f1_score
    return float(grid[int(np.argmax([f1_score(y, prob >= t, zero_division=0) for t in grid]))])
