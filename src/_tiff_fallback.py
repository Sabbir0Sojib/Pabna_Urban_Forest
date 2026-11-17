"""Minimal GeoTIFF reader used only when rasterio is not installed.

Reads the georeference from the GeoTIFF tags and decodes uncompressed, Deflate and LZW strips or
tiles (LZW through Pillow). Install rasterio for normal use; this file exists so the analysis can
run in restricted environments.
"""
import io
import struct

import numpy as np
import tifffile
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def _georef(page):
    tags = {k.name: k.value for k in page.tags.values()}
    sx, sy = tags["ModelPixelScaleTag"][:2]
    tp = tags["ModelTiepointTag"]
    nod = tags.get("GDAL_NODATA")
    try:
        nod = float(str(nod).strip("\x00")) if nod is not None else None
    except ValueError:
        nod = None
    return dict(x0=tp[3] - tp[0] * sx, y0=tp[4] + tp[1] * sy, sx=sx, sy=sy, nodata=nod)


def _plane_tiff(strips, w, h, rps, bps, pred, sfmt, tile=None):
    # minimal little-endian single-band TIFF holding given compressed strips (LZW)
    tags=[]; data=b''.join(strips); n=len(strips)
    hdr=8; ntags=12 if not tile else 13; ifd=hdr; ifd_len=2+ntags*12+4
    arr_off=ifd+ifd_len; so_off=arr_off; sbc_off=so_off+4*n; data_off=sbc_off+4*n
    offs=[]; o=data_off
    for s in strips: offs.append(o); o+=len(s)
    def T(tag,typ,cnt,val): return struct.pack('<HHII',tag,typ,cnt,val) if typ==4 or cnt>1 else struct.pack('<HHIHH',tag,typ,cnt,val,0)
    O=T(273 if not tile else 324,4,n,so_off) if n>1 else T(273 if not tile else 324,4,1,offs[0])
    C=T(279 if not tile else 325,4,n,sbc_off) if n>1 else T(279 if not tile else 325,4,1,len(strips[0]))
    mid=[T(277,3,1,1),T(278,4,1,rps)] if not tile else [T(277,3,1,1),T(322,4,1,tile[0]),T(323,4,1,tile[1])]
    ent=[T(256,4,1,w),T(257,4,1,h),T(258,3,1,bps),T(259,3,1,5),T(262,3,1,1),O]+mid+[C,T(284,3,1,1),T(317,3,1,pred),T(339,3,1,sfmt)]
    ent=sorted(ent,key=lambda e: struct.unpack('<H',e[:2])[0])
    b=b'II'+struct.pack('<HI',42,ifd)+struct.pack('<H',ntags)+b''.join(ent)+struct.pack('<I',0)
    b+=struct.pack('<%dI'%n,*offs)+struct.pack('<%dI'%n,*[len(s) for s in strips])+data
    return b


def read_any(path):
    with tifffile.TiffFile(path) as t:
        p = t.pages[0]
        g = _georef(p)
        try:
            return p.asarray(), g          # uncompressed or Deflate: tifffile decodes these itself
        except Exception:
            pass
        fh = t.filehandle
        spp = p.samplesperpixel
        h, w = p.imagelength, p.imagewidth
        offs, cnts = p.dataoffsets, p.databytecounts
        nplanes = spp if p.planarconfig.value == 2 else 1
        if spp > 1 and nplanes == 1:
            raise NotImplementedError("pixel-interleaved LZW: install rasterio")
        per = len(offs) // nplanes
        out = []
        for k in range(nplanes):
            strips = []
            for i in range(k * per, (k + 1) * per):
                fh.seek(offs[i])
                strips.append(fh.read(cnts[i]))
            tb = _plane_tiff(strips, w, h, min(p.rowsperstrip, h) if p.rowsperstrip else 0, p.bitspersample,
                             p.predictor, p.sampleformat, (p.tilewidth, p.tilelength) if p.is_tiled else None)
            a = np.array(Image.open(io.BytesIO(tb)))
            if p.sampleformat == 3 and a.dtype != np.float32:
                a = a.view(np.float32)
            out.append(a)
    return (np.stack(out) if nplanes > 1 else out[0]), g
