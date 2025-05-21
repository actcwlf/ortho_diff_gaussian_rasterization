

from typing import NamedTuple
import torch.nn as nn
import torch
import diff_gaussian_rasterization_cuda as _C


def rasterize_gaussians(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            means3D,
            means2D,
            sh,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            # raster_settings.projmatrix,
            # raster_settings.tanfovx,
            # raster_settings.tanfovy,
            raster_settings.x_min,
            raster_settings.y_min,
            raster_settings.scale,
            raster_settings.threshold,
            raster_settings.kernel_size,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )



        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.ortho_rasterize_gaussians(*args)
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.ortho_rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp.cuda(), radii, sh.cuda(), geomBuffer,
                              binningBuffer, imgBuffer)
        # return color, radii
        return color, radii, num_rendered

    @staticmethod
    def backward(ctx, grad_out_color, _, __):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors


        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D,
                radii,
                colors_precomp,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                # raster_settings.projmatrix,
                # raster_settings.tanfovx,
                # raster_settings.tanfovy,
                raster_settings.x_min,
                raster_settings.y_min,
                raster_settings.scale,
                raster_settings.threshold,
                raster_settings.kernel_size,
                grad_out_color,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.ortho_rasterize_gaussians_backward(
                    *args)

        else:
            grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.ortho_rasterize_gaussians_backward(
                *args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    # tanfovx: float
    # tanfovy: float
    x_min: float
    y_min: float
    scale: float
    threshold: float
    kernel_size: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    # projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings:  GaussianRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.ortho_mark_visible(
                positions,
                raster_settings.viewmatrix,
                # raster_settings.projmatrix,
                raster_settings.threshold
            )

        return visible

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None, scales=None, rotations=None,
                cov3D_precomp=None):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
                (scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )

    def visible_filter(self, means3D, scales=None, rotations=None, cov3D_precomp=None):

        raster_settings = self.raster_settings

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        with torch.no_grad():
            radii = _C.ortho_rasterize_gaussians_filter(
                means3D,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3D_precomp,
                raster_settings.viewmatrix,
                # raster_settings.projmatrix,
                # raster_settings.tanfovx,
                # raster_settings.tanfovy,
                raster_settings.x_min,
                raster_settings.y_min,
                raster_settings.scale,
                raster_settings.threshold,
                raster_settings.kernel_size,
                raster_settings.image_height,
                raster_settings.image_width,
                raster_settings.prefiltered,
                raster_settings.debug
            )
        return radii




