
#ifndef CUDA_ORTHO_RASTERIZER_H_INCLUDED
#define CUDA_ORTHO_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaOrthoRasterizer
{
	class OrthoRasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			// float* projmatrix,
			const float threshold,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			// const float* projmatrix,
			const float* cam_pos,
			// const float tan_fovx, float tan_fovy,
			const float x_min, 
			const float y_min,
			const float scale,
			const float threshold,
			const bool prefiltered,
			float* out_color,
			int* radii = nullptr,
			bool debug = false);


		static void visible_filter(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int M,
			const int width, int height,
			const float* means3D,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			// const float* projmatrix,
			// const float tan_fovx, float tan_fovy,
			const float x_min, 
			const float y_min,
			const float scale,
			const float threshold,
			const bool prefiltered,
			int* radii,
			bool debug);
		
		
		
		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			// const float* projmatrix,
			const float* campos,
			// const float tan_fovx, float tan_fovy,
			const float x_min, 
			const float y_min,
			const float scale,
			const float threshold,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			bool debug);
	};
};

#endif