# A faster `np.unique`: demo

Compares `np.unique` in NumPy `main` with a build that has the changes in the
`numpy` submodule, on real label data.

    git clone --recurse-submodules <this repository>
    pixi run data      # downloads the data
    pixi run lab       # builds both NumPy variants, then opens demo.ipynb
    pixi run plot      # writes summary.png

`pixi.toml` defines three environments:

| environment | NumPy |
|---|---|
| `main` | upstream `main`, built from source |
| `tables` | the `numpy` submodule: the same commit plus the `np.unique` changes |
| `default` | a release, with JupyterLab; the notebook runs here and starts the other two |

Needs [pixi](https://pixi.sh), git and a C and C++ compiler, on Linux or macOS,
with about 6 GB of disk. The first run compiles NumPy twice, which takes about
ten minutes.

`pixi run data` fetches 260 MB from the publishers named at the end of the
notebook, and the *C. elegans* movies, which come as a 3.4 GB archive from a
slow server (allow an hour). The embryo cells load both movies into 8.6 GB of
shared memory; set `FRAME_STEP = 2` in the notebook to halve that.

After changing the submodule, rebuild with `pixi reinstall -e tables numpy`.
