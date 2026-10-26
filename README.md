# MorphoFlow: Sparse-Supervised Generative Shape Modeling with Adaptive Latent Relevance

Official code for the WACV 2027 paper by Mokshagna Sai Teja Karanam, Tushar Kataria and Shireen Elhabian
(Kahlert School of Computing, Scientific Computing and Imaging Institute, University of Utah).

MorphoFlow learns a generative statistical shape model directly from sparse 2D slice annotations of
3D anatomy. Each shape is a latent code of a neural implicit auto-decoder; the population of latent
codes is modeled by an autoregressive normalizing flow whose base distribution uses automatic
relevance determination (ARD), so the model learns which latent dimensions the population actually
uses.

## Repository layout

```
morphoflow/
  train.py                 two-stage training (auto-decoder, then flow + ARD prior)
  evaluate.py              test-time latent fitting, reconstruction and metrics
  models/implicits.py      implicit auto-decoder (occupancy MLP)
  models/flow_prior.py     autoregressive spline flow with ARD base distribution
  utils/                   data loading, slice sampling, I/O, metrics
  configs/                 default, training, inference and paths configs
  casename_files/          train/test splits used in the paper
```

## Installation

```bash
conda create -n morphoflow python=3.9
conda activate morphoflow
pip install -r requirements.txt
```

The code was tested with PyTorch 2.7 and nflows 0.14. A GPU is recommended.

## Data

Each dataset is a folder of binary label volumes in NIfTI format, one file per case named
`<casename>.nii.gz`, split into a training and a test folder:

```
<data_basedir>/
  train/   <casename>.nii.gz ...
  test/    <casename>.nii.gz ...
```

Volumes must be axis-aligned (the affine may only contain positive spacing and a translation in
LPS coordinates). The case lists in `morphoflow/casename_files/<dataset>/` (`train_cases.txt`,
`test_cases.txt`) give the splits used in the paper for VerSe19, OAI-ZIB, liver, kidney, spleen and
left atrium.

## Configuration

All paths come from a paths config; nothing in the code points to a fixed location. Copy
`morphoflow/configs/paths_config.yml` and edit it:

| Key | Meaning |
| --- | --- |
| `model_basedir` | Trained models are written to / read from `<model_basedir>/<model_name>/` |
| `data_basedir` | Dataset root |
| `train_labels_dirname`, `test_labels_dirname` | Sub-folders of `data_basedir` with training / test volumes |
| `casefiles_basedir` | Folder with `train_cases.txt` and `test_cases.txt` |
| `output_basedir` | Reconstructions and metrics are written here |

Relative paths are resolved from the folder the scripts are run in.

Hyperparameters live in YAML configs. `configs/train_config_default.yml` and
`configs/eval_config_default.yml` document every option and hold the defaults; the configs passed on
the command line only need the values that differ.

### Slice sampling

`slice_mode` (training) and `inference_slice_mode` (evaluation) select which slices are observed:

| Mode | Observed slices |
| --- | --- |
| `parallel` | every `slice_step_size`-th slice along `slice_step_axis` (add `use_thick_slices: True` for thick slices) |
| `orthogonal` | one random foreground slice along each axis |
| `mid_orthogonal` | three orthogonal slices through the foreground center |
| `axial`, `coronal`, `sagittal` | three random foreground slices along axis 2, 1 and 0 respectively |

## Training

Run from the `morphoflow/` folder:

```bash
cd morphoflow
python train.py -p configs/paths_config.yml -c configs/training/train_config_ard.yml
```

Training has two stages:

1. **Burn-in** (`burnin_epochs`): the implicit auto-decoder and one latent code per training shape
   are fitted to the observed slices with a BCE + Dice loss.
2. **Prior fitting** (remaining epochs): the decoder and latent codes are frozen, and the
   autoregressive flow with the ARD base distribution is fitted to the latent codes by maximum
   likelihood, with a Gamma(`gamma_a`, `gamma_b`) hyperprior on the ARD precisions weighted by
   `lat_reg_lambda`.

Checkpoints, TensorBoard logs, `log.txt` and a copy of the training config are written to
`<model_basedir>/<model_name>/`.

## Inference

```bash
python evaluate.py -p configs/paths_config.yml -m thin_ad -c configs/inference/eval_config_ard.yml
```

`-m` is the model folder name inside `model_basedir`; its stored training config is merged with the
evaluation config. For every test case a latent code is initialized with a sample from the learned
prior, optimized for `latent_num_iters` steps to fit the observed slices, and decoded at full
resolution.

Results go to `<output_basedir>/<model_name>_<latent_num_iters>_eval_<slice settings>/`:

- `log.txt`: per-case and mean DSC, ASD, HD95 and HD;
- `<case>_pred.nii.gz`, `<case>_lr.nii.gz`: reconstruction and observed slices;
- `<case>_latent.npy`, `<case>_latent_z0.npy`: fitted latent code and its image under the flow;
- `ard_sigma.npy` and `cum_variance_test_*.png`: learned ARD scales and the number of dimensions
  explaining 95/97/98% of the prior variance.

## Acknowledgments

The implicit auto-decoder and the data pipeline build on the code of Amiranashvili et al.,
[Learning continuous shape priors from sparse data with neural implicit functions](https://github.com/menzelab/implicit-shape-reconstruction)
(Medical Image Analysis, 2024). Surface distance metrics are from
[google-deepmind/surface-distance](https://github.com/google-deepmind/surface-distance). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for their licenses.

## Citation

```bibtex
@inproceedings{karanam2027morphoflow,
  title     = {MorphoFlow: Sparse-Supervised Generative Shape Modeling with Adaptive Latent Relevance},
  author    = {Karanam, Mokshagna Sai Teja and Kataria, Tushar and Elhabian, Shireen},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```
