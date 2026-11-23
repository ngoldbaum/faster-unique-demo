"""Time reading four real label datasets and np.unique on them in both builds, and draw the summary figure.

    pixi run plot              # times both builds (a few minutes), writes summary.png
    pixi run plot --replot     # redraws from the timings of the last run
"""
import json
import os
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
COLORS = {"main": "#9aa0a6", "this PR": "#1a73e8"}
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

call = eval("lambda a: " + sys.argv[2])
paths = []
for path in map(Path, sys.argv[3:]):
    paths += sorted(path.glob("*.tif")) if path.is_dir() else [path]
best = None
for _ in range(int(sys.argv[1])):
    seconds = {"read": 0.0, "unique": 0.0}
    for path in paths:
        t0 = time.perf_counter()
        a = read(path)
        t1 = time.perf_counter()
        call(a)
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
        "analysis": "Nucleus volumes (return_counts)",
        "call": "np.unique(a, return_counts=True)",
        "data": "Cell Tracking Challenge, 385 frames\nuint16, 4.9 billion voxels",
        "source": "doi:10.1038/s41592-023-01879-y",
        "paths": [DATA / "celegans/01", DATA / "celegans/02"],
        "repeats": 1,
    },
    "neurons": {
        "title": "Fly brain neuron segmentation",
        "analysis": "Relabel IDs (return_inverse)",
        "call": "np.unique(a, return_inverse=True)",
        "data": "CREMI sample A, electron microscopy\nuint64, 195 million voxels",
        "repeats": 2,
        "source": "cremi.org",
        "paths": [DATA / "cremi_sample_A.hdf"],
    },
    "atlas": {
        "title": "Mouse brain atlas annotation",
        "analysis": "Region volumes (return_counts)",
        "call": "np.unique(a, return_counts=True)",
        "data": "Allen Mouse Brain CCFv3, 25 µm\nuint32, 77 million voxels",
        "source": "doi:10.1016/j.cell.2020.04.007",
        "paths": [DATA / "allen_annotation_25.nrrd"],
    },
    "land": {
        "title": "Land cover classification",
        "analysis": "Area per class (return_counts)",
        "call": "np.unique(a, return_counts=True)",
        "data": "ESA WorldCover 2021, one tile\nuint8, 1.3 billion pixels",
        "repeats": 2,
        "source": "doi:10.5281/zenodo.7254221",
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
            out = subprocess.run([python, "-c", WORKER, str(workflow.get("repeats", 3)), workflow["call"],
                                  *map(str, workflow["paths"])],
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


def machine():
    """The CPU, memory, operating system and the compiler NumPy was built with."""
    query = ("import json, numpy, platform; c = numpy.show_config(mode='dicts')['Compilers']['c']; "
             "print(json.dumps([c['name'], c['version'], platform.python_version()]))")
    compiler, version, python = json.loads(subprocess.run(
        [BUILDS["this PR"], "-c", query], capture_output=True, text=True, check=True).stdout)
    memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    system = f"macOS {platform.mac_ver()[0]}" if sys.platform == "darwin" else f"Linux {platform.release()}"
    return f"{cpu()}, {memory:.0f} GB, {system}, {compiler} {version}, Python {python}"


def run(replot=False):
    """Time the workflows in both builds, or load the timings of the last run."""
    if replot:
        saved = json.loads(TIMINGS.read_text())
    else:
        saved = {"machine": machine(), "timings": measure()}
        TIMINGS.write_text(json.dumps(saved, indent=1) + "\n")
    return saved


@plt.rc_context({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
def draw(saved):
    """The summary figure for the timings from `run`."""
    timings = saved["timings"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.8), height_ratios=[3.2, 1.5],
                             gridspec_kw={"left": 0.075, "right": 0.985, "top": 0.93, "bottom": 0.16,
                                          "wspace": 0.2, "hspace": 0.62})
    pictures = images()
    for column, (key, workflow) in enumerate(WORKFLOWS.items()):
        top, bottom = axes[0, column], axes[1, column]
        top.imshow(crop(pictures[key]), interpolation="nearest")
        top.axis("off")
        top.set_title(workflow["title"], loc="left", fontsize=13.5, fontweight="bold")
        top.text(0, -0.02, workflow["source"], transform=top.transAxes, va="top", fontsize=8.5, color="#666666")

        seconds = timings[key]
        groups = {"np.unique": {n: seconds[n]["unique"] for n in BUILDS},
                  "with I/O": {n: seconds[n]["read"] + seconds[n]["unique"] for n in BUILDS}}
        for g, values in enumerate(groups.values()):
            for b, (build, value) in enumerate(values.items()):
                y = g * 2.6 + b
                bottom.barh(y, value, height=0.86, color=COLORS[build])
                text = f" {value:.2f} s" if value < 10 else f" {value:.1f} s"
                if build != "main":
                    text += f"   {values['main'] / value:.1f}x faster"
                bottom.text(value, y, text, va="center", fontweight="bold" if build != "main" else "normal")
        bottom.set_yticks([0.5, 3.1], list(groups) if column == 0 else ["", ""])
        bottom.tick_params(axis="y", length=0)
        bottom.invert_yaxis()
        bottom.set_xlim(0, groups["with I/O"]["main"] * 1.5)
        bottom.set_xlabel("seconds")
        bottom.set_title(workflow["analysis"] + "\n" + workflow["data"], loc="left", fontsize=12, color="#444444")
    fig.legend(handles=[Patch(color=COLORS[build], label=build) for build in BUILDS],
               loc="lower center", ncols=2, frameon=False)
    fig.text(0.995, 0.012, saved["machine"],
             ha="right", va="bottom", fontsize=8.5, color="#666666")
    return fig


if __name__ == "__main__":
    figure = draw(run(replot="--replot" in sys.argv))
    figure.savefig(HERE / "summary.png", dpi=300, facecolor="white")
    print("wrote", HERE / "summary.png")
