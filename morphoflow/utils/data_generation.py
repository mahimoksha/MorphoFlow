import time
from enum import IntEnum
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch.utils import data

from . import geometry_utils, io_utils

# Sparsification start IDs for training/validataion/test data
SPRSF_START_MAIN_SEED = 1
# Splitting data into training and validation
SPLIT_TRAIN_VAL_SEED = 2
# Sparsification start IDs for internal train/val in AD
SPRSF_START_SHARED_VAL_SEED = 3


class PhaseType(IntEnum):
    """Training, validation and inference."""
    TRAIN = 0
    VAL = 1
    INF = 2


def load_volumes(volumes_dir: Path, casenames: List[str]) -> Tuple[List[torch.Tensor],
                                                                   List[torch.Tensor]]:
    """Load axis-aligned volumes."""
    volumes: List[np.ndarray] = []
    spacings: List[np.ndarray] = []
    for casename in casenames:
        curr_pattern = f'{casename}.nii*'
        files = list(volumes_dir.glob(curr_pattern))
        if len(files) != 1:
            raise ValueError(f'Exactly one file must fit the pattern:\n'
                             f'{volumes_dir / curr_pattern}')
        volume, affine = io_utils.load_nifti_file(files[0])
        # Currently only transformation matrices with scaling & translation are supported
        if not geometry_utils.is_matrix_scaling_and_transform(affine):
            raise ValueError('Local to global image matrix is supposed to be 4x4, have scaling '
                             'and translation components only, and positive scaling. Instead got\n'
                             f'{affine}')
        spacing = np.diagonal(affine)[:3].copy()
        volumes.append(volume)
        spacings.append(spacing)

    # Convert everything to float32 and/or tensors
    volumes_tnsr = [torch.from_numpy(x).to(torch.float32) for x in volumes]
    spacings_tnsr = [torch.from_numpy(x).to(torch.float32) for x in spacings]
    return volumes_tnsr, spacings_tnsr


def compute_thick_slices(volume: torch.Tensor, axis: int, step_size: int) -> torch.Tensor:
    """Return a new volume (as a copy) with the same dimensions as input, where each slice is an
    average of x neighboring slices from the original volume.
    """
    thick_slices = []
    max_num_slices_below = round(step_size / 2) - 1 if step_size % 2 == 0 else \
        round(step_size / 2 - 0.5)
    for center_slice_id in range(volume.shape[axis]):
        # The end should be exclusive: [start_id, end_id)
        start_id = center_slice_id - max_num_slices_below
        end_id = min(start_id + step_size, volume.shape[axis])
        start_id = max(start_id, 0)  # must be after end computation
        subvolume = torch.index_select(volume, axis, torch.arange(start_id, end_id))
        assert subvolume.shape[axis] == end_id - start_id
        mean_slice = torch.mean(subvolume, dim=axis)
        mean_slice = (mean_slice >= 0.5).to(volume.dtype)
        thick_slices.append(mean_slice)
    volume_thick = torch.stack(thick_slices, axis)
    assert volume_thick.shape == volume.shape
    return volume_thick


def sparsen_volume(volume: torch.Tensor, spacing: torch.Tensor, offset: torch.Tensor, axis: int,
                   slice_step_size: int, slice_start_id: int, use_thick_slices: bool) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a new sparsified dense volume (as a copy), its new spacing and new spatial offset. If
    slice_step_size <= 1, return the original volume (no copy), spacing and spatial offset.
    """
    if use_thick_slices:
        volume = compute_thick_slices(volume, axis, slice_step_size)

    if slice_step_size <= 1:
        return volume, spacing, offset

    slice_ids = torch.arange(slice_start_id, volume.shape[axis], slice_step_size)
    # This creates a copy
    volume_sparse = torch.index_select(volume, axis, slice_ids)
    spacing_sparse = spacing.clone()
    spacing_sparse[axis] *= slice_step_size
    offset_sparse = offset.clone()
    offset_sparse[axis] += slice_start_id * spacing[axis]
    return volume_sparse, spacing_sparse, offset_sparse


def split_train_val_caseids(num_cases: int, val_fraction: float) -> Tuple[List[int], List[int]]:
    """Return (reproducible) training and validation case IDs."""
    if num_cases <= 0:
        raise ValueError('Number of cases must be non-negative.')
    val_fraction = max(min(val_fraction, 1.0), 0.0)
    val_set_size = round(num_cases * val_fraction)
    if val_set_size == 0:
        raise ValueError(f'Empty validation set (fraction {val_fraction} of {num_cases} cases).')
    if val_set_size == num_cases:
        raise ValueError(f'Empty training set (fraction {val_fraction} of {num_cases} cases).')

    generator_val_sampling = torch.Generator().manual_seed(SPLIT_TRAIN_VAL_SEED)
    caseids = torch.randperm(num_cases, generator=generator_val_sampling).tolist()
    caseids_train = caseids[:-val_set_size]
    caseids_valid = caseids[-val_set_size:]
    return caseids_train, caseids_valid


def slice_voxel_ids(shape: torch.Size, axis: int, slice_id: int) -> torch.Tensor:
    """Generate [N, 3] voxel IDs of all voxels in slice `slice_id` along `axis`."""
    axis1 = (axis + 1) % 3
    axis2 = (axis + 2) % 3
    ids1, ids2 = torch.meshgrid([torch.arange(shape[axis1]), torch.arange(shape[axis2])],
                                indexing='ij')
    voxel_ids = torch.empty((ids1.numel(), 3), dtype=torch.int64)
    voxel_ids[:, axis] = slice_id
    voxel_ids[:, axis1] = ids1.flatten()
    voxel_ids[:, axis2] = ids2.flatten()
    return voxel_ids


def generate_mid_slice_ids(label: torch.Tensor) -> torch.Tensor:
    """Generate [N, 3] voxel IDs from three orthogonal slices through the foreground center of
    gravity."""
    individual_voxel_ids = [torch.arange(num_elements) for num_elements in label.shape]
    voxel_ids = torch.stack(torch.meshgrid(individual_voxel_ids, indexing='ij'), -1)
    fg_positions = (voxel_ids * label.unsqueeze(-1)).reshape(-1, 3)
    center = torch.sum(fg_positions, dim=0) / torch.sum(label)
    center = torch.round(center).to(torch.int64)
    return torch.cat([slice_voxel_ids(label.shape, i, int(center[i])) for i in range(3)], 0)


def generate_random_slice_ids(label: torch.Tensor, axis: int) -> torch.Tensor:
    """Generate [N, 3] voxel IDs from one random slice along `axis` that contains foreground."""
    other_axes = tuple(i for i in range(3) if i != axis)
    non_empty_slices = torch.nonzero(torch.sum(label, dim=other_axes)).flatten()
    random_index = torch.randint(low=0, high=len(non_empty_slices), size=(1,))
    return slice_voxel_ids(label.shape, axis, int(non_empty_slices[random_index]))


def generate_random_orthogonal_slice_ids(label: torch.Tensor) -> torch.Tensor:
    """Generate [N, 3] voxel IDs from three random orthogonal slices that contain foreground."""
    return torch.cat([generate_random_slice_ids(label, i) for i in range(3)], 0)


def generate_random_single_axis_slice_ids(label: torch.Tensor, axis: int,
                                          num_slices: int = 3) -> torch.Tensor:
    """Generate [N, 3] voxel IDs from `num_slices` random slices with foreground, all orthogonal
    to `axis` (drawn with replacement)."""
    return torch.cat([generate_random_slice_ids(label, axis) for _ in range(num_slices)], 0)


# Observed slices per case for every mode except 'parallel' (which sparsifies with a step size).
SLICE_ID_GENERATORS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    'orthogonal': generate_random_orthogonal_slice_ids,
    'mid_orthogonal': generate_mid_slice_ids,
    'axial': partial(generate_random_single_axis_slice_ids, axis=2),
    'coronal': partial(generate_random_single_axis_slice_ids, axis=1),
    'sagittal': partial(generate_random_single_axis_slice_ids, axis=0),
}
SLICE_MODES = ('parallel', *SLICE_ID_GENERATORS)


class LabelVolumes(data.Dataset):
    """Full-resolution label volumes, held in memory."""
    def __init__(self, labels_dir: Path, casenames: List[str], verbose: bool):
        self.casenames = casenames
        self.labels, self.spacings = load_volumes(labels_dir, self.casenames)
        # Offsets describe the position of the center of the first voxel. Again, this corresponds to
        # align_corners=False:
        self.offsets = [x / 2 for x in self.spacings]  # type: List[torch.Tensor]

        # Compute max volume size
        # Unlike with typical medical images, we define the bbox size with align_corners=False:
        image_sizes = [torch.tensor(label.shape) * spacing
                       for label, spacing in zip(self.labels, self.spacings)]
        self.image_size: torch.Tensor = torch.stack(image_sizes).max(dim=0)[0]

        if verbose:
            print(f'Volume size: {self.image_size}.')

    def __len__(self):
        return len(self.casenames)


class SliceDataset(LabelVolumes):
    """Observes a fixed set of full slices per case (e.g. three orthogonal slices, or a single
    slice along one axis). The slices are chosen once, when the dataset is created.
    """
    def __init__(self, labels_dir: Path, casenames: List[str],
                 slice_id_generator: Callable[[torch.Tensor], torch.Tensor], verbose: bool):
        super().__init__(labels_dir, casenames, verbose)
        self.fixed_voxel_ids = [slice_id_generator(x) for x in self.labels]

    def __getitem__(self, item: int):
        label = self.labels[item]

        # Extract positions of the observed slices in [N, 1, 1, 3]
        voxel_ids = self.fixed_voxel_ids[item]
        label_values = label[voxel_ids[:, 0], voxel_ids[:, 1], voxel_ids[:, 2]]
        voxel_ids = voxel_ids.unsqueeze(1).unsqueeze(1)
        label_values = label_values.unsqueeze(1).unsqueeze(1)
        spacing = self.spacings[item]
        offset = self.offsets[item]
        coords = voxel_ids * spacing + offset

        # Keys are plural because examples will be combined to batches
        return {
            'coords': coords,
            'labels': label_values,
            'casenames': self.casenames[item],
            'caseids': item,
            'labels_hr': label,
            'spacings_hr': spacing,
            'offsets_hr': offset
        }


class ImplicitDataset(LabelVolumes):
    """For training encoder-free implicit functions on parallel slices. Generated coordinates take
    spacing into account.
    """
    def __init__(self, labels_dir: Path, casenames: List[str],
                 val_fraction: float, num_points_per_example_per_dim: int, slice_step_size: int,
                 slice_step_axis: int, use_thick_slices: bool,
                 do_yield_full_res: bool, phase_type: PhaseType, verbose: bool):
        """
        Dataset for implicit autodecoder.
        :param labels_dir:
        :param casenames: List of casenames to be processed.
        :param val_fraction: fraction of training data to be used for validation.
        :param num_points_per_example_per_dim: number of random points (per dim) per sparse volume
            to be returned. If -1, then use all points for each example.
        :param slice_step_size: sparsification step size; should be at least 2.
        :param slice_step_axis: sparsification direction.
        :param do_yield_full_res: whether to return full resolution GT in addition to sparse labels.
                                  Should be set to false if batch size != 1 in DataLoader (otherwise
                                  batching won't work).
        :param phase_type: training, validation or inference.
        :param verbose:
        """
        super().__init__(labels_dir, casenames, verbose)
        self.do_yield_full_res = do_yield_full_res

        # This is used for as a link between externally stored latent vectors and data examples
        self.caseids = list(range(len(self.casenames)))

        # Calculate sparse volumes
        if slice_step_size <= 1:
            raise ValueError('Step size should be at least 2.')

        gen = torch.Generator().manual_seed(SPRSF_START_MAIN_SEED)
        slice_start_ids = torch.randint(0, slice_step_size, [len(self.labels)],
                                        generator=gen).tolist()
        results = [sparsen_volume(vol, spacing, offset, slice_step_axis, slice_step_size,
                                  slice_start_id, use_thick_slices)
                   for vol, spacing, offset, slice_start_id in
                   zip(self.labels, self.spacings, self.offsets, slice_start_ids)]
        self.labels_sparse = [x[0] for x in results]
        self.spacings_sparse = [x[1] for x in results]
        self.offsets_sparse = [x[2] for x in results]

        # When training and validating, use some of training examples for both training and
        # validation (with different sparsification starts).
        if phase_type != PhaseType.INF:
            # Reproducible validation set selection
            _, val_caseids = split_train_val_caseids(len(self.casenames), val_fraction)
            # Reproducible sparsification start slice id -- different one for train and val
            gen_val_sparsification = torch.Generator().manual_seed(SPRSF_START_SHARED_VAL_SEED)
            start_slice_ids = torch.randint(0, 2, [len(val_caseids)],
                                            generator=gen_val_sparsification).tolist()
            if phase_type == PhaseType.VAL:
                start_slice_ids = [(x + 1) % 2 for x in start_slice_ids]
            # Sparsify the validation cases
            for caseid, start_slice_id in zip(val_caseids, start_slice_ids):
                self.labels_sparse[caseid], self.spacings_sparse[caseid], \
                    self.offsets_sparse[caseid] = \
                    sparsen_volume(self.labels_sparse[caseid], self.spacings_sparse[caseid],
                                   self.offsets_sparse[caseid], slice_step_axis, 2,
                                   start_slice_id, False)
            # Now drop all non-validation cases for validation -- this HAS to happen after both
            # rounds of sparsification for reproducibility of random generators
            if phase_type == PhaseType.VAL:
                self.labels = [self.labels[i] for i in val_caseids]
                self.spacings = [self.spacings[i] for i in val_caseids]
                self.offsets = [self.offsets[i] for i in val_caseids]
                self.labels_sparse = [self.labels_sparse[i] for i in val_caseids]
                self.spacings_sparse = [self.spacings_sparse[i] for i in val_caseids]
                self.offsets_sparse = [self.offsets_sparse[i] for i in val_caseids]
                self.casenames = [self.casenames[i] for i in val_caseids]
                self.caseids = val_caseids

        self.num_points_per_example = num_points_per_example_per_dim ** self.image_size.numel() \
            if num_points_per_example_per_dim > 0 else -1

    def __getitem__(self, item: int):
        label = self.labels_sparse[item]
        # Use sparse (1D) or dense (3D) volumes
        if self.num_points_per_example > 0:  # use random subsample from sparse volume
            # Generate a 1D vector with sparse coords [N, 1, 1, 3]
            ndims = label.ndim
            voxel_ids = torch.empty(self.num_points_per_example, ndims, dtype=torch.int64)
            for i in range(ndims):
                voxel_ids[:, i] = torch.randint(0, label.shape[i], [self.num_points_per_example])
            label_values = label[voxel_ids[:, 0], voxel_ids[:, 1], voxel_ids[:, 2]]
            voxel_ids = voxel_ids.unsqueeze(1).unsqueeze(1)
            label_values = label_values.unsqueeze(1).unsqueeze(1)
        else:  # use full sparse volume
            # Generate a 3D dense volume [SX, SY, SZ, 3]
            individual_voxel_ids = [torch.arange(num_elements) for num_elements in label.shape]
            individual_voxel_ids_meshed = torch.meshgrid(individual_voxel_ids, indexing='ij')
            voxel_ids = torch.stack(individual_voxel_ids_meshed, -1)
            label_values = label

        spacing = self.spacings_sparse[item]
        offset = self.offsets_sparse[item]
        coords = voxel_ids * spacing + offset

        # Keys are plural because examples will be combined to batches
        result_dict = {
            'coords': coords,
            'labels': label_values,
            'spacings': spacing,
            'offsets': offset,
            'casenames': self.casenames[item],
            'caseids': self.caseids[item]
        }
        # Yield full-res data only during inference (where batch size is 1)
        if self.do_yield_full_res:
            result_dict['labels_hr'] = self.labels[item]
            result_dict['spacings_hr'] = self.spacings[item]
            result_dict['offsets_hr'] = self.offsets[item]
        return result_dict


def create_data_loader(params: Dict, phase_type: PhaseType, slice_mode: str,
                       verbose: bool) -> data.DataLoader:
    """Create the data loader for a phase. Training and validation both draw from the training
    cases (the AD validation set is a subset of them, observed with different slices); inference
    uses the test cases. `slice_mode` is one of SLICE_MODES and selects which voxels are observed.
    """
    if phase_type == PhaseType.INF:
        labels_dirname, casefilename = params['test_labels_dirname'], params['test_casefile']
    else:
        labels_dirname, casefilename = params['train_labels_dirname'], params['train_casefile']
    labels_dir = params['data_basedir'] / labels_dirname
    casefiles_dir = params['casefiles_basedir']
    if not labels_dir.exists() or not casefiles_dir.exists():
        raise ValueError('At least one of following data directories does not exist:'
                         f'\n{labels_dir}\n{casefiles_dir}')
    casenames = io_utils.load_casenames(casefiles_dir / casefilename)

    is_training = phase_type == PhaseType.TRAIN
    if is_training:
        data_name = 'training data'
        batch_size = params['batch_size_train']
        num_points_per_example_per_dim = params['num_points_per_example_per_dim_train']
    else:
        # Validation and inference employ full volumes, which may differ in shape between examples
        data_name = 'validation data' if phase_type == PhaseType.VAL else 'test data'
        batch_size = 1
        num_points_per_example_per_dim = -1

    if verbose:
        print(f'Loading {data_name} into memory ({slice_mode} slices)...')
    t0 = time.time()
    ds: LabelVolumes
    if slice_mode == 'parallel':
        # Yield full resolution volumes only during inference (with batch size 1)
        do_yield_full_res = phase_type == PhaseType.INF
        ds = ImplicitDataset(
            labels_dir, casenames, params['val_fraction'], num_points_per_example_per_dim,
            params['slice_step_size'], params['slice_step_axis'], params['use_thick_slices'],
            do_yield_full_res, phase_type, verbose)
    elif slice_mode in SLICE_ID_GENERATORS:
        ds = SliceDataset(labels_dir, casenames, SLICE_ID_GENERATORS[slice_mode], verbose)
    else:
        raise ValueError(f'Unknown slice mode {slice_mode!r}, expected one of {SLICE_MODES}.')
    if verbose:
        print(f'Loading {data_name} done ({len(ds)} images): {time.time() - t0:.2f}s')

    return data.DataLoader(ds, batch_size, shuffle=is_training,
                           num_workers=params['num_workers'], drop_last=is_training)
