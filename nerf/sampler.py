import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        origins = ray_bundle.origins
        directions = ray_bundle.directions

        z_vals = torch.linspace(
            self.min_depth,
            self.max_depth,
            self.n_pts_per_ray,
            device=origins.device
        ).view(1, self.n_pts_per_ray, 1)

        # TODO (Q1.4): Sample points from z values （N_rays, N_pts, 3)
        sample_points = origins.unsqueeze(1) + directions.unsqueeze(1) * z_vals

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )

# q4.2
class CoarseFineRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.n_fine = getattr(cfg, 'n_fine', 64)     # fine
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth
    
    def sample(self, bins, weights):
        weight_norm = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(weight_norm, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        prob = torch.rand(list(cdf.shape[:-1]) + [self.n_fine], device=cdf.device)

        # Inverse Transform Sampling
        index = torch.searchsorted(cdf, prob, right=True)
        min = torch.min((cdf.shape[-1] - 1) * torch.ones_like(index), index)
        max = torch.max(torch.zeros_like(index - 1), index - 1)
        index_range = torch.stack([max, min], -1)

        cdf_range = torch.gather(cdf, -1, index_range)
        bins_range = torch.gather(bins, -1, index_range)

        # Linear Interpolation
        diff = cdf_range[..., 1] - cdf_range[..., 0]
        diff = torch.where(diff < 1e-5, torch.ones_like(diff), diff)
        t = (prob - cdf_range[..., 0]) / diff
        samples = bins_range[..., 0] + t * (bins_range[..., 1] - bins_range[..., 0])

        return samples

    def forward(
        self,
        ray_bundle,
        weights=None
    ):
        origins = ray_bundle.origins
        directions = ray_bundle.directions

        if weights is None:
            # Stratified Sampling
            z_vals = torch.linspace(
                self.min_depth,
                self.max_depth,
                self.n_pts_per_ray,
                device=origins.device
            ).view(1, self.n_pts_per_ray, 1)
        else:
            # Coarse Fine Sampling
            z_vals_coarse = ray_bundle.sample_lengths[..., 0] # (N_rays, N_coarse)
            z_vals_mid = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
            
            z_vals_fine = self.sample(z_vals_mid, weights[..., 1:-1, 0])
            
            # Combine and sort
            z_vals, _ = torch.sort(torch.cat([z_vals_coarse, z_vals_fine], dim=-1), dim=-1)
            z_vals = z_vals.unsqueeze(-1)

        sample_points = origins.unsqueeze(1) + directions.unsqueeze(1) * z_vals

        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler,
    'coarsefine': CoarseFineRaysampler
}