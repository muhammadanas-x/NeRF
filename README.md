# NeRF

This project trains a Neural Radiance Field in PyTorch from Blender images and camera poses. The model uses positional encoding, coarse-to-fine ray sampling, and volume rendering to synthesize new views of a scene. You can render individual RGB and depth images, create an orbit video, or watch frames render in a live viewer.

The model follows the approach introduced by Mildenhall et al. at ECCV 2020. [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)

## Setup

Run the commands below from the repository root in a Python 3 environment. Install PyTorch for your platform, choosing a CUDA build if you have a supported NVIDIA GPU. The NeRF scripts fall back to CPU when CUDA is unavailable, but training and rendering can take much longer. [PyTorch installation guide](https://pytorch.org/get-started/locally/)

```sh
python -m pip install numpy imageio tqdm pillow
```

For MP4 output and the live viewer, install these additional packages. The inference script uses PyAV for orbit videos, and the viewer uses Matplotlib for its preview and ImageIO's FFmpeg backend for video.

```sh
python -m pip install av matplotlib imageio-ffmpeg
```

Run `python test.py` to check whether PyTorch detects CUDA.

## Train a scene

Download the synthetic Blender dataset and extract the `lego` scene into the path shown below, or set `SCENE_DIR` in `nerf_train.py` to the path of another extracted scene. It should point directly to the folder containing `transforms_train.json`. [Original NeRF data release](https://github.com/bmild/nerf#running-code)

```text
nerf_synthetic/nerf_synthetic/lego/
├── transforms_train.json
└── train/
    ├── r_0.png
    └── …
```

```sh
python nerf_train.py
```

Training processes one scene at a time and writes `nerf_weights.pth` when it finishes. The checkpoint stores the model weights, render settings, and training camera poses. You can adjust `MAX_ITERS`, `BATCH_SIZE`, and `HALF_RES` in `nerf_train.py`.

## Render a trained scene

| Command | Output |
| --- | --- |
| `python nerf_infer.py` | A 360° RGB orbit video at `nerf_orbit.mp4` |
| `python nerf_infer.py --mode single --azimuth 45 --elevation 20 --radius 4 --depth` | `nerf_single.png` and `nerf_single_depth.png` |
| `python nerf_infer.py --mode training --n_views 5` | Up to five PNGs from saved training poses in `rendered_views/` |
| `python nerf_viewer.py` | A live RGB and depth preview, with frames and an RGB orbit video saved under `renders/` |

For `nerf_infer.py`, use `--weights` to load a different checkpoint or `--out` to change the output path. Lower `--chunk` if inference runs out of GPU memory. For a headless viewer run, set `SHOW_LIVE = False` in `nerf_viewer.py`.

Use `--depth` with `single` or `training` mode for depth PNGs. Orbit depth export currently passes an unsupported `quality` option to ImageIO's PyAV writer. [ImageIO PyAV writer documentation](https://imageio.readthedocs.io/en/stable/_autosummary/imageio.plugins.pyav.PyAVPlugin.write.html)

Git ignores the downloaded dataset, model checkpoints, and default render outputs.
