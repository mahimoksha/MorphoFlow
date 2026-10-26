from typing import List

import torch


def _block(num_ch_in: int, num_ch_out: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(num_ch_in, num_ch_out),
        torch.nn.ReLU(True)
    )


class OccupancyPredictor(torch.nn.Module):
    """Residual MLP that maps a latent vector and local coordinates to an occupancy logit."""
    def __init__(self, latent_dim: int, spatial_dim: int, num_layers: int,
                 layers_with_coords: List[int], hidden_dim: int = 128):
        super().__init__()

        self.layers_with_coords = layers_with_coords
        in_channels = [hidden_dim] * num_layers
        for lyr_id in self.layers_with_coords:
            if lyr_id == 0:
                in_channels[lyr_id] = latent_dim + spatial_dim
            else:
                in_channels[lyr_id] = hidden_dim + spatial_dim

        self.res_layers = torch.nn.ModuleList(
            [_block(in_channels[i], hidden_dim) for i in range(num_layers - 1)])
        self.last_layer = torch.nn.Linear(in_channels[-1], 1)

    def forward(self, closest_latents: torch.Tensor, local_coords: torch.Tensor) -> torch.Tensor:
        """Inputs have shape [B, *, Z] where Z is latent/spatial dimensionality. Return shape
        [B, *, 1]."""
        features = closest_latents
        for i, layer in enumerate(self.res_layers):
            append_coords = i in self.layers_with_coords
            if append_coords:
                features = torch.cat([features, local_coords], dim=-1)

            out = layer(features)
            # Don't use skip connections if we appended coordinates
            features = out if append_coords else features + out

        features = self.last_layer(features)
        return features


class AutoDecoder(torch.nn.Module):
    """Encoder-free implicit shape prediction."""
    def __init__(self, lat_dim: int, spatial_dim: int, image_size: torch.Tensor,
                 occnet_num_layers: int, occnet_layers_with_coords: List[int]):
        """
        Implicit AutoDecoder.

        :param lat_dim: Dimensionality of the latent vector.
        :param spatial_dim: Number of spatial dimentions.
        :param image_size: Physical size of the image volume.
        :param occnet_num_layers: Number of linear layers in the implicit occupancy MLP.
        :param occnet_layers_with_coords: Layers with concatenated coordinates in occupancy MLP.
        """
        super().__init__()
        self.image_size: torch.Tensor
        self.register_buffer('image_size', image_size)
        # Global coordinates of the latent vector
        latent_coords: torch.Tensor = image_size / 2  # noqa
        self.latent_coords: torch.Tensor
        self.register_buffer('latent_coords', latent_coords)
        self.occp_pred: torch.nn.Module
        self.occp_pred = OccupancyPredictor(lat_dim, spatial_dim, occnet_num_layers,
                                            occnet_layers_with_coords)

    def forward(self, latents: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """Latents have shape [B, Z]. Coordinates are in the global coord. system with shape
        [B, *ST, 3].
        """
        # Compute local coordinates w.r.t. the latent vector position.
        local_coords = coordinates - self.latent_coords
        latents = latents.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        latents = latents.expand(-1, coordinates.shape[1], coordinates.shape[2],
                                 coordinates.shape[3], -1)
        predictions = self.occp_pred(latents, local_coords)
        # Remove the feature dim (MLP, channels-last)
        predictions = predictions.squeeze(4)
        return predictions  # [B, *ST]
