import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10 # epsilon: used for Numerical Stability
    ):
        # TODO (1.5): Compute transmittance using the equation described in the README
        alpha = 1.0 - torch.exp(-rays_density * deltas)
        transmit_fraction = 1.0 - alpha + eps
        # torch.cumprod: Cumulative Product
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(transmit_fraction[:, :1]), transmit_fraction[:, :-1]], dim=1),
            dim=1
        )

        # TODO (1.5): Compute weight used for rendering from transmittance and alpha (w = T * alpha)
        weights = transmittance * alpha
        return weights
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        feature = torch.sum(rays_feature * weights, dim=1)

        return feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        # Q4.2
        is_coarse_fine = isinstance(implicit_fn, (tuple, list))
        if is_coarse_fine:
            coarse_fn, fine_fn = implicit_fn

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            if not is_coarse_fine:
                # Sample points along the ray
                cur_ray_bundle = sampler(cur_ray_bundle)
                n_pts = cur_ray_bundle.sample_shape[1]

                # Call implicit function with sample points
                implicit_output = implicit_fn(cur_ray_bundle)
                density = implicit_output['density']
                feature = implicit_output['feature']

                # Compute length of each ray segment
                depth_values = cur_ray_bundle.sample_lengths[..., 0]
                deltas = torch.cat(
                    (
                        depth_values[..., 1:] - depth_values[..., :-1],
                        1e10 * torch.ones_like(depth_values[..., :1]),
                    ),
                    dim=-1,
                )[..., None]

                # Compute aggregation weights
                weights = self._compute_weights(
                    deltas.view(-1, n_pts, 1),
                    density.view(-1, n_pts, 1)
                ) 

                # TODO (1.5): Render (color) features using weights
                feature = self._aggregate(
                    weights,
                    feature.view(-1, n_pts, feature.shape[-1]) # -1: automatically infer the size of that specific dimension
                )

                # TODO (1.5): Render depth map
                depth = self._aggregate(
                    weights,
                    depth_values.view(-1, n_pts, 1)
                )

                # Return
                cur_out = {
                    'feature': feature,
                    'depth': depth,
                }
            else:
                cur_ray_bundle_coarse = sampler(cur_ray_bundle, weights=None)
                n_pts_course = cur_ray_bundle_coarse.sample_shape[1]

                out_course = coarse_fn(cur_ray_bundle_coarse)
                depth_course = cur_ray_bundle_coarse.sample_lengths[..., 0]
                deltas_course = torch.cat(
                    (
                        depth_course[..., 1:] - depth_course[..., :-1], 
                        1e10 * torch.ones_like(depth_course[..., :1])
                    ), 
                    dim=-1
                )[..., None]

                weights_course = self._compute_weights(
                    deltas_course.view(-1, n_pts_course, 1), 
                    out_course['density'].view(-1, n_pts_course, 1)
                )
                feature_course = self._aggregate(
                    weights_course, 
                    out_course['feature'].view(-1, n_pts_course, out_course['feature'].shape[-1])
                )

                cur_ray_bundle_fine = sampler(cur_ray_bundle, weights=weights_course)
                n_pts_fine = cur_ray_bundle_fine.sample_shape[1]

                out_fine = fine_fn(cur_ray_bundle_fine)
                depth_fine = cur_ray_bundle_fine.sample_lengths[..., 0]
                deltas_fine = torch.cat(
                    (
                        depth_fine[..., 1:] - depth_fine[..., :-1], 
                        1e10 * torch.ones_like(depth_fine[..., :1])
                    ), 
                    dim=-1
                )[..., None]

                weights_fine = self._compute_weights(
                    deltas_fine.view(-1, n_pts_fine, 1), 
                    out_fine['density'].view(-1, n_pts_fine, 1)
                )
                feature_fine = self._aggregate(
                    weights_fine, 
                    out_fine['feature'].view(-1, n_pts_fine, out_fine['feature'].shape[-1])
                )
                depth_fine_map = self._aggregate(
                    weights_fine, 
                    depth_fine.view(-1, n_pts_fine, 1)
                )

                cur_out = {
                    'feature': feature_fine,
                    'feature_coarse': feature_course,
                    'depth': depth_fine_map
                }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


def sdf_to_density(signed_distance, alpha, beta):
    # TODO (Q4.3): Convert signed distance to density with alpha, beta parameters
    pass

class VolumeSDFRenderer(VolumeRenderer):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False
        self.alpha = cfg.alpha
        self.beta = cfg.beta

        self.cfg = cfg

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            distance, color = implicit_fn.get_distance_color(cur_ray_bundle.sample_points)
            density = None # TODO (Q4.3): convert SDF to density

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            geometry_color = torch.zeros_like(color)

            # Compute color
            color = self._aggregate(
                weights,
                color.view(-1, n_pts, color.shape[-1])
            )

            # Return
            cur_out = {
                'feature': color,
                "geometry": geometry_color
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer,
    'volume_sdf': VolumeSDFRenderer
}
