
#ifndef CUDA_ORTHO_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_ORTHO_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace ORTHO_FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		// const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		// const float focal_x, float focal_y,
		// const float tan_fovx, float tan_fovy,
		const float x_min, 
		const float y_min,
		const float scale,
		const float threshold,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color);
	void filter_preprocess(int P, int M,
		const float* means3D,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* cov3D_precomp,
		const float* viewmatrix,
		// const float* projmatrix,
		const int W, int H,
		// const float focal_x, float focal_y,
		// const float tan_fovx, float tan_fovy,
		const float x_min, 
		const float y_min,
		const float scale,
		const float threshold,
		int* radii,
		float* cov3Ds,
		const dim3 grid,
		bool prefiltered);
}


#endif