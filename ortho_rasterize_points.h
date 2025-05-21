
#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
	
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
OrthoRasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	// const torch::Tensor& projmatrix,
	const float x_min, 
	const float y_min,
	const float scale,
	const float threshold,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 OrthoRasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    // const torch::Tensor& projmatrix,
	// const float tan_fovx, 
	// const float tan_fovy,
	const float x_min, 
	const float y_min,
	const float scale,
	const float threshold,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug);
		
torch::Tensor OrthoMarkVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		// torch::Tensor& projmatrix,
		const float threshold
		);


torch::Tensor
OrthoRasterizeGaussiansfilterCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,  // 1
	const torch::Tensor& cov3D_precomp, // None, scales+rotation 和conv3D_precomp二者有其一即可
	const torch::Tensor& viewmatrix,
	// const torch::Tensor& projmatrix,
	// const float tan_fovx, 
	// const float tan_fovy,
	const float x_min, 
	const float y_min,
	const float scale,
	const float threshold,
    const int image_height,
    const int image_width,
	const bool prefiltered,  // false
	const bool debug);