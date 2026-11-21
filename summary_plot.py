"""Time reading four real label datasets and np.unique on them in both builds, and draw the summary figure.

    pixi run plot              # times both builds (a few minutes), writes summary.png
    pixi run plot --replot     # redraws from the timings of the last run
"""
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib import colormaps
from matplotlib.patches import Patch

HERE = Path(__file__).parent
DATA = HERE / "data"
BUILDS = {"main": HERE / ".pixi/envs/main/bin/python", "this PR": HERE / ".pixi/envs/tables/bin/python"}
COLORS = {"main": "#5f6368", "this PR": "#1a73e8"}
READ_COLOR = "#d5d8dc"
TIMINGS = HERE / "summary-timings.json"

# Runs in each build: read a dataset from the file its publisher distributes,
# then call np.unique, timing the two steps separately.
WORKER = r"""
import gzip, json, sys, time
from pathlib import Path
import numpy as np

def read(path):
    if path.suffix == ".hdf":
        import h5py
        with h5py.File(path) as f:
            return f["volumes/labels/neuron_ids"][...]
    if path.suffix == ".nrrd":
        raw = path.read_bytes()
        return np.frombuffer(gzip.decompress(raw[raw.index(b"\n\n") + 2:]), dtype="<u4")
    import tifffile
    return tifffile.imread(path)

paths = []
for path in map(Path, sys.argv[2:]):
    paths += sorted(path.glob("*.tif")) if path.is_dir() else [path]
best = None
for _ in range(int(sys.argv[1])):
    seconds = {"read": 0.0, "unique": 0.0}
    for path in paths:
        t0 = time.perf_counter()
        a = read(path)
        t1 = time.perf_counter()
        np.unique(a)
        seconds["read"] += t1 - t0
        seconds["unique"] += time.perf_counter() - t1
        del a
    if best is None or sum(seconds.values()) < sum(best.values()):
        best = seconds
print(json.dumps(best))
"""

WORKFLOWS = {
    "embryo": {
        "title": "3D C. elegans cell tracking",
        "data": "Cell Tracking Challenge, 385 frames\nuint16, 4.9 billion voxels",
        "paths": [DATA / "celegans/01", DATA / "celegans/02"],
        "repeats": 1,
    },
    "neurons": {
        "title": "Fly brain neuron segmentation",
        "data": "CREMI sample A, electron microscopy\nuint64, 195 million voxels",
        "paths": [DATA / "cremi_sample_A.hdf"],
    },
    "atlas": {
        "title": "Mouse brain atlas annotation",
        "data": "Allen Mouse Brain CCFv3, 25 µm\nuint32, 77 million voxels",
        "paths": [DATA / "allen_annotation_25.nrrd"],
    },
    "land": {
        "title": "Land cover classification",
        "data": "ESA WorldCover 2021, one tile\nuint8, 1.3 billion pixels",
        "paths": [DATA / "worldcover_N39W108.tif"],
    },
}


def label_image(labels):
    """One colour per label, background black."""
    _, codes = np.unique(labels, return_inverse=True)
    rgb = colormaps["tab20"](codes.reshape(labels.shape) * 7919 % 20)[..., :3]
    rgb[labels == 0] = 0
    return rgb


def crop(image, aspect=4 / 3):
    """The centre of an image, cut to a common width to height ratio."""
    height, width = image.shape[:2]
    new_height, new_width = min(height, round(width / aspect)), min(width, round(height * aspect))
    top, left = (height - new_height) // 2, (width - new_width) // 2
    return image[top:top + new_height, left:left + new_width]


def images():
    embryo = tifffile.imread(DATA / "celegans/01/man_seg150.tif").max(axis=0)
    neurons = np.load(DATA / "cremi_neuron_ids.npy", mmap_mode="r")[62]
    raw = np.load(DATA / "cremi_raw_slice.npy")[..., None] / 255
    atlas = np.load(DATA / "allen_annotation_25.npy", mmap_mode="r")
    tile = np.load(DATA / "worldcover_N39W108.npy", mmap_mode="r")[::40, ::40]
    classes = {10: "#006400", 20: "#ffbb22", 30: "#ffff4c", 40: "#f096ff", 50: "#fa0000", 60: "#b4b4b4",
               70: "#f0f0f0", 80: "#0064c8", 90: "#0096a0", 95: "#00cf75", 100: "#fae6a0"}
    lut = np.zeros((256, 3))
    for code, colour in classes.items():
        lut[code] = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return {
        "embryo": label_image(embryo),
        "neurons": 0.45 * raw + 0.55 * label_image(np.asarray(neurons)),
        "atlas": label_image(np.asarray(atlas[:, :, atlas.shape[2] // 2 - 40].T)),
        "land": lut[np.asarray(tile)],
    }


def measure():
    timings = {}
    for key, workflow in WORKFLOWS.items():
        timings[key] = {}
        for build, python in BUILDS.items():
            out = subprocess.run([python, "-c", WORKER, str(workflow.get("repeats", 3)), *map(str, workflow["paths"])],
                                 capture_output=True, text=True, check=True)
            timings[key][build] = json.loads(out.stdout)
            print(f"{key:8s} {build:8s} read {timings[key][build]['read']:7.3f} s   "
                  f"np.unique {timings[key][build]['unique']:7.3f} s", flush=True)
    return timings


def cpu():
    if sys.platform == "darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.machine()


if "--replot" in sys.argv:
    saved = json.loads(TIMINGS.read_text())
else:
    saved = {"cpu": cpu(), "timings": measure()}
    TIMINGS.write_text(json.dumps(saved, indent=1) + "\n")
timings = saved["timings"]

plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 4, figsize=(16, 6.3), height_ratios=[3.2, 1], layout="constrained")
pictures = images()
for column, (key, workflow) in enumerate(WORKFLOWS.items()):
    top, bottom = axes[0, column], axes[1, column]
    top.imshow(crop(pictures[key]), interpolation="nearest")
    top.axis("off")
    top.set_title(workflow["title"], loc="left", fontsize=13.5, fontweight="bold")

    seconds = timings[key]
    names = list(BUILDS)
    total = {n: seconds[n]["read"] + seconds[n]["unique"] for n in names}
    bottom.barh(names, [seconds[n]["read"] for n in names], color=READ_COLOR)
    bottom.barh(names, [seconds[n]["unique"] for n in names], left=[seconds[n]["read"] for n in names],
                color=[COLORS[n] for n in names])
    bottom.invert_yaxis()
    bottom.set_xlim(0, total["main"] * 1.22)
    faster = seconds["main"]["unique"] / seconds["this PR"]["unique"]
    bottom.set_xlabel("seconds")
    bottom.set_title(workflow["data"] + "\n\n", loc="left", fontsize=12, color="#444444")
    bottom.text(0, 1.06, f"np.unique: {faster:.1f}x faster\nwhole task: {total['main'] / total['this PR']:.1f}x faster",
                transform=bottom.transAxes, fontsize=13, fontweight="bold", color=COLORS["this PR"])
    for i, n in enumerate(names):
        bottom.text(total[n], i, f" {total[n]:.2f} s" if total[n] < 10 else f" {total[n]:.1f} s", va="center")
fig.suptitle(f"Read a label file and list its labels: main against this PR   ({saved['cpu']})",
             x=0.01, ha="left", fontsize=17)
fig.legend(handles=[Patch(color=READ_COLOR, label="read and decode the file"),
                    Patch(color=COLORS["main"], label="np.unique, main"),
                    Patch(color=COLORS["this PR"], label="np.unique, this PR")],
           loc="outside lower center", ncols=3, frameon=False)
fig.savefig(HERE / "summary.png", dpi=150, facecolor="white")
print("wrote", HERE / "summary.png")
