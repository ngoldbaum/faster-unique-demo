"""Download the label data the demo uses from its publishers and unpack it into data/.

About 260 MB of downloads plus the C. elegans movies, which come as a 3.4 GB
archive from a slow server (allow an hour). Unpacked, data/ takes 3.5 GB.
Files that are already there are skipped.
"""
import gzip
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import h5py
import numpy as np
import tifffile

DATA = Path(__file__).parent / "data"
SOURCES = {
    # neuron segmentation of fly brain electron microscopy, https://cremi.org
    "cremi_sample_A.hdf": "https://cremi.org/static/data/sample_A_20160501.hdf",
    # Allen Mouse Brain CCFv3 annotation, 25 um, https://atlas.brain-map.org
    "allen_annotation_25.nrrd": "https://download.alleninstitute.org/informatics-archive/"
                                "current-release/mouse_ccf/annotation/ccf_2017/annotation_25.nrrd",
    # ESA WorldCover 2021 land cover, one 3 x 3 degree tile, https://esa-worldcover.org
    "worldcover_N39W108.tif": "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                              "ESA_WorldCover_10m_2021_v200_N39W108_Map.tif",
}
# Cell Tracking Challenge Fluo-N3DH-CE, https://celltrackingchallenge.net
CELEGANS = "http://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-CE.zip"


def fetch(name, url):
    target = DATA / name
    if target.exists():
        return target
    part = target.with_name(name + ".part")
    with urllib.request.urlopen(url) as response, open(part, "wb") as out:
        total, done = int(response.headers.get("Content-Length", 0)), 0
        while block := response.read(1 << 20):
            out.write(block)
            done += len(block)
            print(f"\r{name}: {done >> 20} of {total >> 20} MB", end="", file=sys.stderr)
    print(file=sys.stderr)
    part.rename(target)
    return target


DATA.mkdir(exist_ok=True)
for name, url in SOURCES.items():
    fetch(name, url)

if not (DATA / "celegans/02").is_dir():
    archive = fetch("Fluo-N3DH-CE.zip", CELEGANS)
    with zipfile.ZipFile(archive) as z:
        for movie in ("01", "02"):
            folder = DATA / "celegans" / movie
            folder.mkdir(parents=True, exist_ok=True)
            for member in z.namelist():
                if member.startswith(f"Fluo-N3DH-CE/{movie}_ST/SEG/") and member.endswith(".tif"):
                    with z.open(member) as src, open(folder / Path(member).name, "wb") as out:
                        shutil.copyfileobj(src, out)
    archive.unlink()

if not (DATA / "cremi_neuron_ids.npy").exists():
    with h5py.File(DATA / "cremi_sample_A.hdf") as f:
        np.save(DATA / "cremi_neuron_ids.npy", f["volumes/labels/neuron_ids"][...])
        np.save(DATA / "cremi_raw_slice.npy", f["volumes/raw"][62])
if not (DATA / "allen_annotation_25.npy").exists():
    raw = (DATA / "allen_annotation_25.nrrd").read_bytes()
    end = raw.index(b"\n\n")
    sizes = [int(v) for v in raw[:end].decode().split("sizes:")[1].splitlines()[0].split()]
    volume = np.frombuffer(gzip.decompress(raw[end + 2:]), dtype="<u4").reshape(sizes[::-1])
    np.save(DATA / "allen_annotation_25.npy", volume)
if not (DATA / "worldcover_N39W108.npy").exists():
    np.save(DATA / "worldcover_N39W108.npy", tifffile.imread(DATA / "worldcover_N39W108.tif"))

for name in ("cremi_neuron_ids", "allen_annotation_25", "worldcover_N39W108"):
    a = np.load(DATA / f"{name}.npy", mmap_mode="r")
    print(f"{name}: {a.shape} {a.dtype}")
for movie in ("01", "02"):
    print(f"celegans/{movie}: {len(list((DATA / 'celegans' / movie).glob('*.tif')))} frames")
