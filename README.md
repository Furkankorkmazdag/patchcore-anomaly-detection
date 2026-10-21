# PatchCore Anomaly Detection

A compact, ROS-free computer-vision project for unsupervised anomaly detection using [Anomalib](https://github.com/openvinotoolkit/anomalib)'s PatchCore implementation.

The repository provides four workflows:

1. Extract video frames into an image dataset.
2. Train PatchCore using **normal images only**.
3. Evaluate scores on held-out normal and anomalous images, then calibrate an image-level threshold.
4. Run live anomaly detection from camera index `0` (typically `/dev/video0` on Linux).

Videos, datasets, model checkpoints, exported weights, and generated results are not included. Add your own files locally; large artifacts are excluded from Git by `.gitignore`.

## Project structure

```text
patchcore-anomaly-detection/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── scripts/
│   ├── common.py
│   ├── extract_frames.py
│   ├── train_patchcore.py
│   ├── analyze_results.py
│   └── detect_camera.py
├── videos/                   # Local input videos (not committed)
├── dataset/
│   ├── train/normal/         # Normal images used for training
│   └── test/
│       ├── normal/           # Held-out normal images for threshold calibration
│       └── anomaly/          # Images containing anomalies
└── results/                  # Checkpoints, exported model, reports, and plots
```

## 1. Set up the environment

Python 3.10 or 3.11 is recommended. From the repository root on Ubuntu:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For NVIDIA GPU acceleration, install a PyTorch build compatible with the local NVIDIA driver and CUDA setup before running the scripts. Verify the installation with:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The training and analysis scripts use CUDA when available and otherwise fall back to CPU. The first training run may download pretrained ImageNet backbone weights.

## 2. Extract frames from video

Put local input videos in `videos/`. The example filenames below describe the clip contents: `anomaly_absent.mp4` contains no anomalies and `anomaly_present.mp4` contains anomalies. These videos are not included in the repository.

Extract one frame per second from a normal clip for training:

```bash
python scripts/extract_frames.py \
  --video videos/anomaly_absent.mp4 \
  --output dataset/train/normal \
  --every-seconds 1 \
  --clear
```

Extract frames from a clip containing anomalies for evaluation:

```bash
python scripts/extract_frames.py \
  --video videos/anomaly_present.mp4 \
  --output dataset/test/anomaly \
  --every-seconds 1 \
  --clear
```

For `dataset/test/normal/`, use a separate normal recording when possible. If you only have one normal clip, extract a different time segment rather than copying the training frames. Adjust the timestamps to fit the length of your video:

```bash
python scripts/extract_frames.py \
  --video videos/anomaly_absent.mp4 \
  --output dataset/test/normal \
  --start-seconds 60 \
  --duration-seconds 20 \
  --every-seconds 1 \
  --clear
```

Useful options:

- `--every-seconds 0` saves every frame.
- `--start-seconds` chooses the start timestamp.
- `--duration-seconds` limits the clip segment; `0` means until the end.
- `--max-frames 500` limits the number of saved frames.
- `--clear` removes existing supported image files from the output directory before extraction.

**Dataset quality matters:** keep training images free of anomalies. Separate training and evaluation by recording or time segment; placing near-identical frames in both splits can inflate evaluation results. Use footage representative of the camera, lighting, angle, and environment expected during deployment.

## 3. Train PatchCore

```bash
python scripts/train_patchcore.py
```

Optional settings:

```bash
python scripts/train_patchcore.py --batch-size 8 --num-workers 2 --normal-split 0.2
```

PatchCore learns a memory bank from normal training images. The script creates an Anomalib checkpoint and exports a Torch model for live inference under `results/patchcore/exported/`.

Do not upload checkpoints or exported weights to GitHub unless you explicitly intend to publish them. These artifacts are excluded by `.gitignore`.

## 4. Analyze model results and calibrate a threshold

Make sure both directories contain held-out images:

- `dataset/test/normal/`
- `dataset/test/anomaly/`

Then run:

```bash
python scripts/analyze_results.py
```

The script writes these files to `results/analysis/`:

- `prediction_scores.csv` — image label, predicted label, score, and threshold.
- `score_distribution.png` — normal/anomaly score histograms.
- `summary.json` — image-level evaluation metrics.
- `threshold.json` — threshold used by the live camera script.

By default, the threshold is the 99th percentile of the **held-out normal** image scores. This is a starting point, not a universal threshold. Review the score distribution and false-positive rate, and validate the threshold against separate footage representative of the deployment environment. The reported metrics are image-level metrics; boxes shown in the camera view are approximate regions derived from the anomaly map, not separately trained object-detection boxes.

You can select a checkpoint explicitly:

```bash
python scripts/analyze_results.py --checkpoint /path/to/model.ckpt
```

## 5. Detect anomalies with a live camera

Connect the camera and run:

```bash
python scripts/detect_camera.py --camera 0
```

On Linux, camera index `0` commonly maps to `/dev/video0`. Use `--camera 1` for another camera index. Press **Q** or **Esc** to close the preview window.

To save the annotated live feed to an MP4 file:

```bash
python scripts/detect_camera.py \
  --camera 0 \
  --model results/patchcore/exported/patchcore.pt \
  --threshold-file results/analysis/threshold.json \
  --save-video results/annotated_camera.mp4
```

You can also pass a manually calibrated threshold without a JSON file:

```bash
python scripts/detect_camera.py --camera 0 --threshold 0.22
```

The value `0.22` is an example only. Use the threshold obtained for your own model and validation data. If the exported model is missing, run training first. On Ubuntu, camera access problems may require checking `/dev/video0` permissions and closing other applications using the camera.

The live script uses Anomalib's exported Torch model format and opts in to loading trusted exported model code. Only load model files that you created yourself or otherwise trust.

## Notes and limitations

- The image anomaly score determines whether a frame is flagged. The anomaly-map percentile is used only to visualize high-response regions after a frame is flagged.
- Contour boxes are approximate localization aids and are not guaranteed to match exact object boundaries.
- Camera motion, reflections, exposure changes, edges, shadows, and lighting differences can produce false positives. Train on representative normal footage and evaluate using separate clips.
- For reproducible experiments, record the Anomalib, PyTorch, CUDA, GPU, and threshold details.

## License

MIT. See `LICENSE`.
