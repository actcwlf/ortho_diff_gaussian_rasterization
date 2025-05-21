/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * Modified by Xiang Liu
 */

#include <torch/extension.h>
#include "rasterize_points.h"
#include "ortho_rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("rasterize_aussians_filter", &RasterizeGaussiansfilterCUDA);
  m.def("mark_visible", &markVisible);

  m.def("ortho_rasterize_gaussians", &OrthoRasterizeGaussiansCUDA);
  m.def("ortho_rasterize_gaussians_backward", &OrthoRasterizeGaussiansBackwardCUDA);
  m.def("ortho_rasterize_gaussians_filter", &OrthoRasterizeGaussiansfilterCUDA);
  m.def("ortho_mark_visible", &OrthoMarkVisible);
  
}