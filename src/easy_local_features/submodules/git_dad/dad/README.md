<p align="center">
  <h1 align="center"> <ins>DaD:</ins> Distilled Reinforcement Learning for Diverse Keypoint Detection</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=Ul-vMR0AAAAJ">Johan Edstedt</a>
    ·
    <a href="https://scholar.google.com/citations?user=FUE3Wd0AAAAJ">Georg Bökman</a>
    ·
    <a href="https://scholar.google.com/citations?user=6WRQpCQAAAAJ">Mårten Wadenbäck</a>
    ·
    <a href="https://scholar.google.com/citations?user=lkWfR08AAAAJ">Michael Felsberg</a>
  </p>
  <h2 align="center"><p>
    <a href="https://arxiv.org/abs/2503.07347" align="center">Paper</a>
  </p></h2>
  <p align="center">
      <img src="assets/qualitative.jpg" alt="example" width=80%>
      <br>
      <em>DaD's a pretty good keypoint detector, probably the best.</em>
  </p>
</p>
<p align="center">
</p>

## Run
```python
import dad
from PIL import Image
img_path = "assets/0015_A.jpg"
W, H = Image.open(img_path).size# your image shape,
detector = dad.load_DaD()
detections = detector.detect_from_path(
  img_path, 
  num_keypoints = 512,
  return_dense_probs=True)
detections["keypoints"] # 1 x 512 x 2, normalized coordinates of keypoints
detector.to_pixel_coords(detections["keypoints"], H, W)
detections["keypoint_probs"] # 1 x 512, probs of sampled keypoints
detections["dense_probs"] # 1 x H x W, probability map
```

## Visualize
```python
import dad
from dad.utils import visualize_keypoints
detector = dad.load_DaD()
img_path = "assets/0015_A.jpg"
vis_path = "vis/0015_A_dad.jpg"
visualize_keypoints(img_path, vis_path, detector, num_keypoints = 512)
```

## Install
Get uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
### In an existing env
Assuming you already have some env active:
```bash
uv pip install dad@git+https://github.com/Parskatt/dad.git
```
### As a project 
For dev, etc:
```bash
git clone git@github.com:Parskatt/dad.git
uv sync
source .venv/bin/activate
```

## Evaluation
For to evaluate, e.g., DaD on ScanNet1500 with 512 keypoints, run
```bash
python experiments/benchmark.py --detector DaD --num_keypoints 512 --benchmark ScanNet1500
```
Note: leaving out num_keypoints will run the benchmark for all numbers of keypoints, i.e., [512, 1024, 2048, 4096, 8192].
### Third party detectors
We provide wrappers for a somewhat large set of previous detectors, 
```bash
python experiments/benchmark.py --help
```

## Training
To train our final model from the emergent light and dark detector, run
```bash
python experiments/repro_paper_results/distill.py
```
The emergent models come from running
```bash
python experiments/repro_paper_results/rl.py
```
Note however that the types of detectors that come from this type of training is stochastic, and you may need to do several runs to get a detector that matches our results.

## Changes versus paper version
1. I've updated inference to not include border pixels, see https://github.com/Parskatt/dad/issues/3 for details. However, training is still including borders. Results on most benchmarks is similar.

## How I run experiments
(Note: You don't have to do this, it's just how I do it.)
At the start of a new day I typically run 
```bash
python new_day.py
```
This creates a new folder in experiments, e.g., `experiments/w11/monday`.
I then typically just copy the contents of a previous experiment, e.g.,
```bash
cp experiments/repro_paper_results/rl.py experiments/w11/monday/new-cool-hparams.py
```
Change whatever you want to change in `experiments/w11/monday/new-cool-hparams.py`.

Then run it with
```bash
python experiments/w11/monday/new-cool-hparams.py
```
This will be tracked in wandb as `w11-monday-new-cool-hparams` in the `DaD` project.

You might not want to track stuff, and perhaps display some debugstuff, then you can run instead as, which also won't log to wandb
```bash
DEBUG=1 python experiments/w11/monday/new-cool-hparams.py
```
## Evaluation Results
<details>
<summary>MegaDepth1500 Essential</summary>

| Method ↓ | Max № Keypoints |||||
|----------|------|------|------|------|------|
|          | 512  | 1024 | 2048 | 4096 | 8192 |
| DaD (Paper) | 64.9 | 67.9 | 69.4 | 69.3 | 68.8 |
| DaD (This Repo) | 64.8 | 68.0 | 69.4 | 69.6 | 68.8 |
</details>

<details>
<summary>MegaDepth1500 Fundamental</summary>

| Method ↓ | Max № Keypoints |||||
|----------|------|------|------|------|------|
|          | 512  | 1024 | 2048 | 4096 | 8192 |
| DaD (Paper) | 50.6 | 55.5 | 57.4 | 57.8 | 56.4 |
| DaD (This Repo) | 50.6 | 55.5 | 57.7 | 58.2 | 56.4 |
</details>

<details>
<summary>ScanNet1500 Essential</summary>

| Method ↓ | Max № Keypoints |||||
|----------|------|------|------|------|------|
|          | 512  | 1024 | 2048 | 4096 | 8192 |
| DaD (Paper) | 25.7 | 27.9 | 28.3 | 28.9 | 29.3 |
| DaD (This Repo) | 26.3 | 28.0 | 28.6 | 29.0 | 29.2 |
</details>

<details>
<summary>ScanNet1500 Fundamental</summary>

| Method ↓ | Max № Keypoints |||||
|----------|------|------|------|------|------|
|          | 512  | 1024 | 2048 | 4096 | 8192 |
| DaD (Paper) | 18.3 | 20.6 | 21.7 | 22.2 | 23.0 |
| DaD (This Repo) | 18.6 | 21.4 | 21.9 | 22.5 | 23.3 |
</details>

<details>
<summary>HPatches Viewpoint</summary>

| Method ↓ | Max № Keypoints |||||
|----------|------|------|------|------|------|
|          | 512  | 1024 | 2048 | 4096 | 8192 |
| DaD (Paper) | 58.2 | 60.5 | 61.0 | 61.5 | 61.4 |
| DaD (This Repo) | 58.3 | 60.4 | 61.0 | 61.2 | 61.5 |
</details>


## Licenses
DaD is MIT licensed.

Third party detectors in [dad/detectors/third_party](dad/detectors/third_party) have their own licenses. If you use them, please refer to their respective licenses in [here](licenses) (NOTE: There may be more licenses you need to care about than the ones listed. Before using any third party code, make sure you're following their respective license).

## BibTeX

```txt
@article{edstedt2025dad,
  title={{DaD: Distilled Reinforcement Learning for Diverse Keypoint Detection}},
  author={Edstedt, Johan and B{\"o}kman, Georg and Wadenb{\"a}ck, M{\aa}rten and Felsberg, Michael},
  journal={arXiv preprint arXiv:2503.07347},
  year={2025}
}
```
