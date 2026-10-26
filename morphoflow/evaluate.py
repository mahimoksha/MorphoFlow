"""Reconstruct the test shapes with a trained MorphoFlow model.

For every test case a latent is drawn from the learned prior (ARD base sample pushed through the
inverse flow), optimized to fit the observed slices, and decoded at full resolution. Reports DSC /
ASD / HD95 / HD and the effective dimensionality of the ARD prior.

Usage: python evaluate.py -p configs/paths_config.yml -m thin_ad \
           -c configs/inference/eval_config_ard.yml
"""
import sys
import time
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from models import flow_prior, implicits  # noqa: E402
from train import create_flow, create_loss, create_model  # noqa: E402
from utils import config_io, data_generation, impl_utils, io_utils, nn_utils  # noqa: E402

# Learning rate for test-time latent optimization
LATENT_LR = 1e-2


def optimize_latents(net: implicits.AutoDecoder, latents_batch: torch.Tensor,
                     labels: torch.Tensor, coords: torch.Tensor, lr: float, num_iters: int,
                     device: torch.device, max_num_const_train_dsc: int, verbose: bool) -> None:
    """Optimize latent vectors for a single example.
    max_num_const_train_dsc: if train dice doesn't change this number of times, stop training. -1
                             means never stop early.
    """
    criterion = create_loss().to(device)
    optimizer_val = torch.optim.Adam([latents_batch], lr=lr)

    eval_every_x_steps = 10
    print_every_x_evals = 10
    prev_train_dsc = 0.0
    num_const_train_dsc = 0

    net.eval()

    t0 = time.time()
    for i in range(num_iters):
        labels_pred = net(latents_batch, coords)
        loss = criterion(labels_pred, labels)
        optimizer_val.zero_grad()
        loss.backward()
        optimizer_val.step()
        if (i + 1) % eval_every_x_steps == 0:
            dsc = nn_utils.dice_coeff(torch.sigmoid(labels_pred), labels, 0.5).item()
            if verbose and round((i + 1) / eval_every_x_steps) % print_every_x_evals == 0:
                print(f'Step {i + 1:04d}/{num_iters:04d}: loss {loss.item():.4f} DSC {dsc:.3f} '
                      f'({time.time() - t0:.1f}s)')
                t0 = time.time()
            if round(dsc, 3) == round(prev_train_dsc, 3):
                num_const_train_dsc += 1
            else:
                num_const_train_dsc = 0
            if num_const_train_dsc == max_num_const_train_dsc:
                print(f'Reached stopping critertion after {i + 1} steps. '
                      f'Optimization has converged.')
                break
            prev_train_dsc = dsc


def export_batch(batch: dict, labels_pred: np.ndarray, spacings: np.ndarray, target_dir: Path,
                 latents_batch: np.ndarray, latents_batch_z0: np.ndarray):
    """Write the prediction, the observed (sparse) input, the fitted latent z and its image z0
    under the flow for every case."""
    # Create a sparse volume with hi-res spacing/offset
    spacings_lr = batch['spacings_hr']
    offsets_lr = batch['offsets_hr']
    label_values: torch.Tensor = batch['labels'].flatten(start_dim=1)
    voxel_ids = (batch['coords'] - offsets_lr) / spacings_lr
    voxel_ids = torch.round(voxel_ids).to(torch.int64).flatten(start_dim=1, end_dim=3)
    labels_lr = torch.zeros_like(batch['labels_hr'])
    labels_lr[:, voxel_ids[:, :, 0], voxel_ids[:, :, 1], voxel_ids[:, :, 2]] = label_values
    labels_lr = labels_lr.numpy()
    spacings_lr = spacings_lr.numpy()

    for i in range(labels_pred.shape[0]):
        casename = batch['casenames'][i]
        io_utils.save_nifti_file(labels_pred[i, 0].astype(np.uint8), np.diag([*spacings[i], 1]),
                                 target_dir / f'{casename}_pred.nii.gz')
        # Sparse volumes with hi-res spacing/offset are aligned with the hi-res volume
        io_utils.save_nifti_file(labels_lr[i].astype(np.uint8), np.diag([*spacings_lr[i], 1]),
                                 target_dir / f'{casename}_lr.nii.gz')
        np.save(target_dir / f'{casename}_latent.npy', latents_batch[i])
        np.save(target_dir / f'{casename}_latent_z0.npy', latents_batch_z0[i])


def report_ard_prior(log_sigmas: torch.Tensor, latents_train: torch.Tensor, target_dir: Path):
    """Print the effective dimensionality of the ARD prior, plot its cumulative variance and store
    the prior scales and training latent statistics."""
    k95, k97, k98 = [flow_prior.num_effective_dims(log_sigmas, f) for f in (0.95, 0.97, 0.98)]
    print(f'Effective dimensions: k95 {k95}, k97 {k97}, k98 {k98}')

    sigma2_sorted = torch.sort(torch.exp(log_sigmas.cpu()) ** 2, descending=True)[0]
    cum_ratio = torch.cumsum(sigma2_sorted, dim=0) / sigma2_sorted.sum()
    plt.figure()
    plt.plot(cum_ratio.numpy())
    plt.axhline(0.95, linestyle='--')
    plt.title('Cumulative Variance Explained')
    plt.savefig(target_dir / f'cum_variance_test_{k95}_95%_{k97}_97%_{k98}_98%.png', dpi=200)
    plt.close()

    np.save(target_dir / 'ard_sigma.npy', torch.exp(log_sigmas.cpu()).numpy())
    np.save(target_dir / 'train_latents_mean.npy', latents_train.mean(dim=0).cpu().numpy())
    np.save(target_dir / 'train_latents_std.npy', latents_train.std(dim=0).cpu().numpy())
    mean_sq_length = torch.mean(torch.sum(torch.square(latents_train), dim=1))
    print(f'Train latents avg length squared: {mean_sq_length}')


def main():
    params = config_io.parse_config_eval()
    export_predictions = params['export_predictions']
    latent_num_iters = params['latent_num_iters']
    latent_dim = params['latent_dim']
    model_dir = params['model_basedir'] / params['model_name']
    slice_mode = params['inference_slice_mode']

    if slice_mode == 'parallel':
        target_dirname = (f'{params["model_name"]}_{latent_num_iters}_eval'
                          f'_ax{params["slice_step_axis"]}_x{params["slice_step_size"]}')
    else:
        target_dirname = f'{params["model_name"]}_{latent_num_iters}_eval_{slice_mode}'
    target_dir = params['output_basedir'] / target_dirname
    target_dir.mkdir(parents=True, exist_ok=params['allow_overwriting'])
    print(f'Writing results to: {target_dir}')
    # Redirect stdout to file + stdout
    sys.stdout = io_utils.Logger(target_dir / 'log.txt', 'w')

    print(f'Evaluating model \'{params["model_name"]}\'.')
    if slice_mode == 'parallel':
        print(f'Reconstructing from slices with step size {params["slice_step_size"]} '
              f'along axis {params["slice_step_axis"]}.')
    else:
        print(f'Reconstructing from {slice_mode} slices.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds_loader = data_generation.create_data_loader(params, data_generation.PhaseType.INF,
                                                   slice_mode, True)
    # The image size is overwritten by the one stored with the model
    net = create_model(params, torch.ones(3, dtype=torch.float32))
    flow = create_flow(params)
    model_state = io_utils.load_latest_checkpoint(model_dir, 'checkpoint', 'pth', True)[0]
    if 'net' not in model_state or 'flow' not in model_state \
            or 'latents_train' not in model_state:
        raise ValueError('Incompatible model state in stored checkpoint.')
    net.load_state_dict(model_state['net'], strict=True)
    flow.load_state_dict(model_state['flow'], strict=True)
    net = net.to(device).eval()
    flow = flow.to(device).eval()

    log_sigmas = model_state['flow']['_distribution.log_scale'].to(device)
    report_ard_prior(log_sigmas, model_state['latents_train'].detach(), target_dir)

    # Check that all volumes lie inside the training image_size
    image_size_train = net.image_size.detach().cpu()
    image_size_curr = ds_loader.dataset.image_size
    # Allow some epsilon
    eps = 1e-2
    if torch.any(image_size_curr > image_size_train + eps):
        # This may not necessarily be a problem, but worth looking into if it happens
        raise ValueError(f'Max image size is larger than current model\'s: '
                         f'{image_size_curr} > {image_size_train} with epsilon {eps}.')

    all_dice_metrics: List[float] = []
    asds: List[float] = []
    hd95s: List[float] = []
    max_distances: List[float] = []

    for i, batch in enumerate(ds_loader):
        t0 = time.time()
        print(f'Batch {i + 1}/{len(ds_loader)}')
        print(f'Batch cases: {batch["casenames"]}')
        labels_gt_sparse = batch['labels'].to(device)
        labels_gt = batch['labels_hr']  # no need to move to GPU
        spacings = batch['spacings_hr']

        # Initialize the latent with a sample from the learned prior: z = f^-1(eps * sigma)
        eps = torch.randn(labels_gt_sparse.shape[0], latent_dim, device=device)
        with torch.no_grad():
            latents_batch = flow._transform.inverse(eps * torch.exp(log_sigmas))[0]
        latents_batch = latents_batch.detach().requires_grad_(True)

        coords = batch['coords'].to(device)
        optimize_latents(net, latents_batch, labels_gt_sparse, coords, LATENT_LR,
                         latent_num_iters, device, params['max_num_const_train_dsc'], True)
        print(f'L2^2(z): {torch.mean(torch.sum(torch.square(latents_batch), dim=1)):.2f}')

        # Full resolution prediction
        target_spatial_shape = torch.tensor(labels_gt.shape[1:])
        labels_pred = impl_utils.sample_latents(latents_batch, net, target_spatial_shape, spacings)
        with torch.no_grad():
            latents_batch_z0 = flow._transform.forward(latents_batch)[0]
        print(f'Batch prediction time: {time.time() - t0:.1f}s')

        # Sigmoid + threshold; add channel dims
        labels_pred_cpu = torch.sigmoid(labels_pred).gt(0.5).unsqueeze(1).numpy()
        labels_gt_cpu = labels_gt.to(torch.bool).unsqueeze(1).numpy()
        spacings_cpu = spacings.numpy()

        impl_utils.eval_batch(labels_pred_cpu, labels_gt_cpu, spacings_cpu, all_dice_metrics,
                              asds, hd95s, max_distances, True)
        if export_predictions:
            export_batch(batch, labels_pred_cpu, spacings_cpu, target_dir,
                         latents_batch.detach().cpu().numpy(),
                         latents_batch_z0.cpu().numpy())

    print('\n')
    for name, values in [('ASD', asds), ('HSD95', hd95s), ('HSD', max_distances),
                         ('DSC', all_dice_metrics)]:
        print(f'{name}: {np.mean(values):.2f} +- {np.std(values):.2f} in '
              f'[{np.min(values):.2f}, {np.max(values):.2f}]')


if __name__ == '__main__':
    main()
