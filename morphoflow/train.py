"""Train MorphoFlow in two stages:

1. Burn-in (epochs < burnin_epochs): fit the implicit auto-decoder and the training latents to the
   observed slices with a BCE + Dice loss.
2. Prior fitting (remaining epochs): freeze the decoder and the latents and fit the autoregressive
   flow with the ARD base distribution to the learned latents (negative log-likelihood plus a Gamma
   hyperprior on the ARD precisions).

Usage: python train.py -p configs/paths_config.yml -c configs/training/train_config_ard.yml
"""
import math
import sys
import time
from typing import Optional

import torch
from torch.optim import optimizer as opt
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from models import flow_prior, implicits
from utils import config_io, data_generation, io_utils, nn_utils


def create_model(params: dict, image_size: torch.Tensor) -> implicits.AutoDecoder:
    return implicits.AutoDecoder(params['latent_dim'], len(image_size), image_size,
                                 params['op_num_layers'], params['op_coord_layers'])


def create_flow(params: dict) -> torch.nn.Module:
    return flow_prior.build_ar_spline_flow_with_ard(params['latent_dim'],
                                                    params['flow_num_layers'])


def create_loss() -> torch.nn.Module:
    return nn_utils.BCEWithDiceLoss('mean', 1.0)


def train_one_epoch(ds_loader: data.DataLoader, net: torch.nn.Module, flow: torch.nn.Module,
                    latents: torch.nn.Parameter, lat_reg_lambda: float, burnin_epochs: int,
                    gamma_a: float, gamma_b: float, optimizer: opt.Optimizer,
                    criterion: torch.nn.Module, metric: torch.nn.Module, device: torch.device,
                    epoch: int, num_epochs_target: int, global_step: torch.Tensor,
                    log_epoch_count: int, logger: Optional[SummaryWriter], verbose: bool):
    loss_running = 0.0
    loss_prior_running = 0.0
    loss_flow_running = 0.0
    loss_dice_running = 0.0
    lat_reg_running = 0.0
    num_losses = 0
    metric_running = 0.0
    num_metrics = 0
    t0 = time.time()

    is_burnin = epoch < burnin_epochs
    if is_burnin:
        net.train()
    else:
        # Stage 2: decoder and latents stay fixed, only the prior is fitted
        net.eval()
        for param in net.parameters():
            param.requires_grad = False
        latents.requires_grad = False
    flow.train()

    for batch in ds_loader:
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        latents_batch = latents[batch['caseids']].to(device)
        coords = batch['coords'].to(device)

        loss_prior = -flow.log_prob(latents_batch).mean()
        lat_reg = flow_prior.ard_gamma_hyperprior(flow._distribution.log_scale, gamma_a, gamma_b)
        labels_pred = net(latents_batch, coords)
        dice_loss = criterion(labels_pred, labels)

        loss_flow = loss_prior
        if lat_reg_lambda > 0:
            # Gradually build up for the first 100 epochs (follows DeepSDF)
            loss_flow = loss_flow + min(1.0, epoch / 100) * lat_reg_lambda * lat_reg

        loss = dice_loss if is_burnin else loss_flow
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            flow._distribution.log_scale.clamp_(min=flow_prior.MIN_LOG_SCALE)

        loss_running += loss.item()
        loss_prior_running += loss_prior.item()
        loss_flow_running += loss_flow.item()
        loss_dice_running += dice_loss.item()
        lat_reg_running += lat_reg.item()
        num_losses += 1
        # Metric returns the sum
        metric_running += metric(labels_pred, labels).item()
        num_metrics += batch['labels'].shape[0]
        global_step += 1

    if epoch % log_epoch_count == 0:
        loss_avg = loss_running / num_losses
        loss_prior_avg = loss_prior_running / num_losses
        loss_flow_avg = loss_flow_running / num_losses
        loss_dice_avg = loss_dice_running / num_losses
        lat_reg_avg = lat_reg_running / num_losses
        metric_avg = metric_running / num_metrics
        num_epochs_trained = epoch + 1

        if logger is not None:
            logger.add_scalar('loss', loss_avg, global_step=num_epochs_trained)
            logger.add_scalar('loss_prior', loss_prior_avg, global_step=num_epochs_trained)
            logger.add_scalar('loss_flow_avg', loss_flow_avg, global_step=num_epochs_trained)
            logger.add_scalar('loss_dice_loss_avg', loss_dice_avg, global_step=num_epochs_trained)
            logger.add_scalar('metric/train', metric_avg, global_step=num_epochs_trained)
            logger.add_scalar('lat_norm', lat_reg.item(), global_step=num_epochs_trained)

        if verbose:
            print(f'[{num_epochs_trained}/{num_epochs_target}] '
                  f'Avg loss: {loss_avg:.4f}; '
                  f'prior loss: {loss_prior_avg:.4f}; '
                  f'flow loss: {loss_flow_avg:.4f}; '
                  f'ARD hyperprior: {lat_reg_avg:.5f}; '
                  f'dice loss: {loss_dice_avg:.4f}; '
                  f'metric: {metric_avg:.3f}; '
                  f'global step nb. {global_step} '
                  f'({time.time() - t0:.1f}s)')
            print('sigma: ', torch.exp(flow._distribution.log_scale).detach().cpu().numpy())


def validate(ds_loader: data.DataLoader, net: torch.nn.Module, flow: torch.nn.Module,
             latents: torch.nn.Parameter, gamma_a: float, gamma_b: float,
             metric: torch.nn.Module, device: torch.device, epoch: int,
             logger: Optional[SummaryWriter], verbose: bool):
    metric_running = 0.0
    num_metrics = 0
    loss_prior_running = 0.0

    t0 = time.time()
    net.eval()
    flow.eval()
    with torch.no_grad():
        log_scale = flow._distribution.log_scale
        lat_reg = flow_prior.ard_gamma_hyperprior(log_scale, gamma_a, gamma_b).item()
        for batch in ds_loader:
            labels = batch['labels'].to(device)
            latents_batch = latents[batch['caseids']].to(device)
            coords = batch['coords'].to(device)

            loss_prior_running += -flow.log_prob(latents_batch).mean().item()
            labels_pred = net(latents_batch, coords)
            # Metric returns the sum
            metric_running += metric(labels_pred, labels).item()
            num_metrics += labels.shape[0]

    metric_avg = metric_running / num_metrics
    loss_prior_avg = loss_prior_running / num_metrics

    if logger is not None:
        logger.add_scalar('metric/val', metric_avg, global_step=(epoch + 1))
        logger.add_scalar('val_loss_prior', loss_prior_avg, global_step=(epoch + 1))
        logger.add_scalar('loss_reg_reg', lat_reg, global_step=(epoch + 1))

    if verbose:
        sigma = torch.exp(log_scale.detach().cpu())
        print(f'[val] metric: {metric_avg:.4f} | prior: {loss_prior_avg:.4f} | '
              f'reg: {lat_reg:.4f} ({time.time() - t0:.1f}s)')
        print(f'sigma mean/min/max: {sigma.mean().item():.4f} / {sigma.min().item():.4f} / '
              f'{sigma.max().item():.4f}; '
              f'k95: {flow_prior.num_effective_dims(log_scale, 0.95)}')


def main():
    params, config_filepath = config_io.parse_config_train()
    model_dir = params['model_basedir'] / params['model_name']
    num_epochs_target = params['num_epochs']
    log_epoch_count = params['log_epoch_count']
    checkpoint_epoch_count = params['checkpoint_epoch_count']
    # Network and flow learning rates scale with the batch size
    learning_rate = params['learning_rate'] * params['batch_size_train']
    latent_dim = params['latent_dim']

    if model_dir.exists():
        raise ValueError('Model directory already exists. Exiting to prevent accidental '
                         f'overwriting.\n{model_dir}')
    model_dir.mkdir(parents=True)
    # The training config is stored with the model; evaluate.py reads it from there
    config_io.write_config(config_filepath, model_dir)
    # Redirect stdout to file + stdout
    sys.stdout = io_utils.Logger(model_dir / 'log.txt', 'a')
    writer = SummaryWriter(log_dir=str(model_dir))
    checkpoint_writer = io_utils.RollingCheckpointWriter(model_dir, 'checkpoint',
                                                         params['max_num_checkpoints'], 'pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print('Warning: no GPU available; training on CPU.')

    slice_mode = params['slice_mode']
    ds_loader_train = data_generation.create_data_loader(
        params, data_generation.PhaseType.TRAIN, slice_mode, True)
    ds_loader_val = data_generation.create_data_loader(
        params, data_generation.PhaseType.VAL, slice_mode, True)
    if len(ds_loader_train) == 0:
        raise ValueError('Number of training examples is smaller than the batch size.')

    net = create_model(params, ds_loader_train.dataset.image_size).to(device)
    flow = create_flow(params).to(device)
    print(net)

    num_examples_train = len(ds_loader_train.dataset)  # type: ignore[arg-type]
    # Initialization scaling follows DeepSDF
    latents_train = torch.nn.Parameter(
        torch.normal(0.0, 1 / math.sqrt(latent_dim), [num_examples_train, latent_dim],
                     device=device))
    optimizer = torch.optim.Adam([
        {'params': net.parameters(), 'lr': learning_rate},
        {'params': flow.parameters(), 'lr': learning_rate},
        {'params': [latents_train], 'lr': params['learning_rate_lat']},
    ])
    criterion = create_loss().to(device)
    metric = nn_utils.DiceLoss(0.5, 'sum', True).to(device)

    # This is a tensor so that it is mutable within other functions
    global_step = torch.tensor(0, dtype=torch.int64)
    for epoch in range(num_epochs_target):
        train_one_epoch(ds_loader_train, net, flow, latents_train, params['lat_reg_lambda'],
                        params['burnin_epochs'], params['gamma_a'], params['gamma_b'],
                        optimizer, criterion, metric, device, epoch, num_epochs_target,
                        global_step, log_epoch_count, writer, True)
        if epoch % log_epoch_count == 0:
            validate(ds_loader_val, net, flow, latents_train, params['gamma_a'],
                     params['gamma_b'], metric, device, epoch, writer, True)
        if epoch % checkpoint_epoch_count == 0:
            checkpoint_writer.write_rolling_checkpoint(
                {'net': net.state_dict(), 'flow': flow.state_dict(),
                 'latents_train': latents_train},
                optimizer.state_dict(), int(global_step.item()), epoch + 1)

    checkpoint_writer.write_rolling_checkpoint(
        {'net': net.state_dict(), 'flow': flow.state_dict(), 'latents_train': latents_train},
        optimizer.state_dict(), int(global_step.item()), num_epochs_target)
    writer.close()


if __name__ == '__main__':
    main()
