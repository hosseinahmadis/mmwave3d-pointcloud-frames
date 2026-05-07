# mmWave Radar Human-Activity Dataset

A labelled mmWave radar point-cloud dataset for human-activity recognition, recorded with a TI AWR1843 / IWR6843 sensor.  
Each sample is a sequence of radar frames stored as NumPy arrays in `new_frames/<activity>/frames.npy`.

---

## Repository layout

```
├── explore_new_frames.ipynb   # exploration, visualisation, and H5 export notebook
├── radar_utils.py             # fps_sample and pad_or_crop helpers
└── README.md
```

The raw frame data is hosted separately on Google Drive (see **Data** below).

---

## Requirements

```
python >= 3.9
numpy
matplotlib
h5py          # only needed for H5 export (Section 6 of the notebook)
```

Install with:

```bash
pip install numpy matplotlib h5py
```

---

## Data

The dataset is hosted on Google Drive as a single ZIP file.

**Download link:** [new_frames.zip — Google Drive](https://drive.google.com/file/d/15rQVJGBP2az3pPDOogL0kc_vKRtU86dN/view?usp=sharing)

### Setup steps

**1. Download** `new_frames.zip` from the link above.

**2. Unzip** it into the same folder as the notebook:

```bash
unzip new_frames.zip
```

After unzipping, the folder structure should look like this:

```
your_folder/
├── explore_new_frames.ipynb
├── radar_utils.py
├── README.md
└── new_frames/
    ├── walk/
    │   └── frames.npy
    ├── run/
    │   └── frames.npy
    ├── sit/
    │   └── frames.npy
    └── ...
```

**3. Open** `explore_new_frames.ipynb` in Jupyter and follow the instructions in the next section.

---

## Using the notebook

### Step 1 — Set your configuration (Cell 1)

The very first code cell contains all the settings you need to adjust. Open it and edit the values to match your needs:

```python
NEW_FRAMES_DIR = "new_frames"   # path to the unzipped data folder
                                 # use an absolute path if the folder is elsewhere,
                                 # e.g. "/home/user/data/new_frames"

N_POINTS       = 64             # number of points per frame after down-sampling
                                 # lower = faster, higher = more spatial detail
                                 # typical values: 32, 64, 128, 256

STACKING_SIZE  = 5              # how many consecutive frames are stacked into
                                 # one training sample (temporal context window)
                                 # set to 1 to disable stacking

STRIDE         = None           # sliding-window stride for stacking
                                 # None  → no overlap (stride = STACKING_SIZE)
                                 # 1     → maximum overlap (one frame shift)
                                 # any integer between 1 and STACKING_SIZE

USE_FPS        = True           # down-sampling method:
                                 # True  → Farthest Point Sampling (better spatial
                                 #         coverage, recommended for training)
                                 # False → intensity crop / zero-pad (much faster,
                                 #         good for quick exploration)

H5_FEATURES    = "x,y,z,intensity,speed"   # feature names — do not change unless
                                             # your frames.npy has a different layout
```

### Step 2 — Run the exploration cells

Cells 1–5 give you:
- A summary table of all activities (frame counts, file sizes).
- Per-feature statistics (min, max, mean, std) over all real points.
- 3-D scatter plots — one frame per activity, coloured by intensity.
- A side-by-side comparison of `pad_or_crop` vs `fps_sample` on the same frame.
- Per-activity window counts for different stride values.

**Example — FPS down-sampling and frame stacking (Cell 4 & 5 output):**

![FPS down-sampling and stacking](assets/image2.png)

### Step 3 — (Optional) Export to H5

**Cell 6** packages everything into a single `.h5` file ready for model training.  
Before running it, set these three extra variables at the top of that cell:

```python
H5_OUT_PATH = "dataset.h5"        # where to save the output file
H5_SPLIT    = (0.75, 0.05, 0.20)  # train / val / test split fractions
H5_SEED     = 42                   # random seed (for reproducibility)
```

Then run Cell 6. It will:
1. Slide a window of `STACKING_SIZE` frames (with the chosen `STRIDE`) over every activity.
2. Down-sample each frame to `N_POINTS` using the method set in Cell 1.
3. Stack frames into samples of shape `(STACKING_SIZE × N_POINTS, 5)`.
4. Randomly assign each sample to train / val / test.
5. Save everything as a compressed `.h5` file.

The output H5 schema is:

```
/points      float32  (N_samples, STACKING_SIZE × N_POINTS, 5)
/class_id    int32    (N_samples,)
/sample_id   int32    (N_samples,)
/class_name  bytes    (N_samples,)
/split       bytes    (N_samples,)    "train" | "val" | "test"
```

---

## Data format

Each `frames.npy` array has shape `(F, N, 5)` where:

| Axis | Meaning |
|------|---------|
| `F`  | Number of radar frames for this activity |
| `N`  | Raw points per frame (variable density; zero-rows = no reflection) |
| `5`  | Feature columns: **x, y, z** (metres), **intensity**, **Doppler speed** (m/s) |

Real points have `intensity > 0`; zero-padded rows have all columns equal to 0.

**Example — raw point-cloud samples (one frame per activity):**

![Raw point-cloud samples](assets/image1.png)

---

## Utility functions (`radar_utils.py`)

### `fps_sample(points, n, seed=0)`
Farthest Point Sampling — selects `n` geometrically spread points from a `(N, 5)` frame.  
Guarantees good spatial coverage; recommended for model training.

### `pad_or_crop(points, n)`
Fast alternative — keeps the `n` highest-intensity points, or zero-pads if the frame is sparse.  
Roughly 10–50× faster than FPS; useful for quick data inspection.

Both functions accept `(N, 5) float32` arrays and return exactly `(n, 5) float32`.

```python
from radar_utils import fps_sample, pad_or_crop

frame_ds = fps_sample(frame, n=64)   # (64, 5)
```

---

## Activities

The dataset contains **20 activity classes** with **99,800 frames** in total (~1.9 GB uncompressed).

| # | Activity | Frames | Shape | Size (MB) |
|---|----------|--------|-------|-----------|
| 0 | Bowing (D) | 5,000 | (5000, 1024, 5) | 97.7 |
| 1 | Lunge (toward left side) | 5,000 | (5000, 1024, 5) | 97.7 |
| 2 | Lunge (toward left-front) | 5,000 | (5000, 1024, 5) | 97.7 |
| 3 | Lunge (toward right side) | 5,000 | (5000, 1024, 5) | 97.7 |
| 4 | Lunge (toward right-front) | 5,000 | (5000, 1024, 5) | 97.7 |
| 5 | Mark time | 5,000 | (5000, 1024, 5) | 97.7 |
| 6 | Picking Up Things | 5,000 | (5000, 1024, 5) | 97.7 |
| 7 | Raising Hand Left | 5,000 | (5000, 1024, 5) | 97.7 |
| 8 | Raising Hand Right | 5,000 | (5000, 1024, 5) | 97.7 |
| 9 | Sitting | 5,000 | (5000, 1024, 5) | 97.7 |
| 10 | Squat | 5,000 | (5000, 1024, 5) | 97.7 |
| 11 | Standing | 5,000 | (5000, 1024, 5) | 97.7 |
| 12 | Stretching and relaxing | 5,000 | (5000, 1024, 5) | 97.7 |
| 13 | Throwing (toward left side) | 4,800 | (4800, 1024, 5) | 93.8 |
| 14 | Throwing (toward right side) | 5,000 | (5000, 1024, 5) | 97.7 |
| 15 | Twist Right | 5,000 | (5000, 1024, 5) | 97.7 |
| 16 | Walking Upright | 5,000 | (5000, 1024, 5) | 97.7 |
| 17 | Walking perpendicularly | 5,000 | (5000, 1024, 5) | 97.7 |
| 18 | Waving Hand Left | 5,000 | (5000, 1024, 5) | 97.7 |
| 19 | Waving Hand Right | 5,000 | (5000, 1024, 5) | 97.7 |
| | **TOTAL** | **99,800** | | **1,949.2** |

---

## Sensor configuration

### Hardware

| Parameter | Value |
|-----------|-------|
| Sensor | TI AWR1843 / IWR6843 |
| Transmit antennas (TX) | 3 |
| Receive antennas (RX) | 4 |
| Virtual antennas | 12 |
| LVDS lanes | 2 |

### Chirp / waveform

| Parameter | Value |
|-----------|-------|
| Start frequency | 60.00 GHz |
| Chirp slope | 50.006 MHz/µs |
| Ramp end time | 60 µs |
| ADC sample rate | 6,000 ksps |
| ADC samples per chirp | 256 |
| Chirps per frame (loop count) | 128 |
| Frame period | 100 ms |
| **Frame rate** | **10 Hz** |

### Derived performance

| Parameter | Value |
|-----------|-------|
| Wavelength | 5.00 mm |
| Bandwidth | 3.000 GHz |
| **Range resolution** | **5.00 cm** |
| Maximum range | ~18.0 m |
| **Velocity resolution** | **0.041 m/s** |
| Maximum velocity | ±2.60 m/s |

### Processing

| Parameter | Value |
|-----------|-------|
| Range FFT size | 2048 |
| Doppler FFT size | 128 |
| Angle FFT size | 180 |
| FFT window | Blackman-Harris |
| Azimuth FoV | ±60° |
| Height filter | −0.5 m to +67 m |
| FFT peaks per antenna group (`n_points_fft`) | 64 |
| Raw points saved per frame | 1,024 (64 peaks × 16 antenna combos) |
| Points per frame in H5 (`n_points`) | 64 (via FPS or intensity crop) |
| Stacking size | 5 frames |
| Stride | 1 (maximum overlap) |
| Points per training sample | 320 (64 pts × 5 frames) |

---

## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{your_paper_doi,
  title   = {[Paper title to be added]},
  author  = {[Authors]},
  journal = {[Venue]},
  year    = {[Year]},
  doi     = {[DOI — to be added upon publication]}
}
```

> **DOI will be added here once the paper is published.**  
> In the meantime, please cite this repository directly.

---

## License

_(Choose a license — e.g. CC BY 4.0 for data, MIT for code)_

---

## Contact

For questions or issues, please open a GitHub Issue or contact _(your email / lab page)_.