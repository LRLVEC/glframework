#include <CUDA/_CUDA_Fractal.h>
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_gl_interop.h>
#include <CUDA/helper_math.h>
#include <cuda/std/complex>

// detailed impl of glue

using complex = cuda::std::complex<float>;

__device__ float mandelbrotFractalKernel(float2 c, int iter)
{
	float c2 = dot(c, c);
	// skip computation inside M1 - https://iquilezles.org/articles/mset1bulb
	if (256.f * c2 * c2 - 96.0f * c2 + 32.0f * c.x - 3.0f < 0.0f) return 0.f;
	// skip computation inside M2 - https://iquilezles.org/articles/mset2bulb
	if (16.f * (c2 + 2.f * c.x + 1.f) - 1.f < 0.f) return 0.f;

	float B = iter / 2;//256.f;
	float l = 0.f;
	float2 z = make_float2(0.f);
	for (int i = 0; i < iter; i++)
	{
		z = make_float2(z.x * z.x - z.y * z.y, 2.f * z.x * z.y) + c;
		if (dot(z, z) > (B * B)) break;
		l += 1.f;
	}
	if (l > iter - 1) return 0.f;

	float sl = l - __log2f(__log2f(dot(z, z))) + 4.f;
	return sl;
}

// 32 * 32 block
__global__ void runMandelbrotFractal(cudaSurfaceObject_t img, int2 _size, float2 _center, float scale, int iter)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < _size.x && y < _size.y)
	{
		float3 col = make_float3(0.f);
#define AA 2
		for (int m = 0; m < AA; m++)
			for (int n = 0; n < AA; n++)
			{
				float2 p = _center + make_float2(x - _size.x / 2 + m / float(AA), _size.y / 2 - y + n / float(AA)) / (float(_size.x) * scale);
				float l = 3.f + mandelbrotFractalKernel(p, iter) * 0.15f;

				col += 0.5f * (1.f + make_float3(__cosf(l + 0.f), __cosf(l + 0.6f), __cosf(l + 1.f)));
			}
		col /= float(AA * AA);
		surf2Dwrite(make_float4(col, 1.f), img, x * sizeof(float4), y);
	}
}

MandelbrotFractalCUDA_Glue::MandelbrotFractalCUDA_Glue(OpenGL::MandelbrotFractalData* _fractalData, GLuint _texture)
	:
	fractalData(_fractalData)
{
	cudaStreamCreate(&stream);
	if (_texture)
	{
		resize(_texture);
	}
}

MandelbrotFractalCUDA_Glue::~MandelbrotFractalCUDA_Glue()
{
	close();
	cudaStreamDestroy(stream);
}

void MandelbrotFractalCUDA_Glue::resize(GLuint _texture)
{
	cudaGraphicsGLRegisterImage(&graphicsResources, _texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
}

void MandelbrotFractalCUDA_Glue::run()
{
	cudaStreamSynchronize(stream);
	cudaGraphicsMapResources(1, &graphicsResources, stream);
	cudaGraphicsSubResourceGetMappedArray(&imgArray, graphicsResources, 0, 0);

	cudaResourceDesc resourceDesc;
	memset(&resourceDesc, 0, sizeof(resourceDesc));
	resourceDesc.resType = cudaResourceTypeArray;
	resourceDesc.res.array.array = imgArray;
	cudaCreateSurfaceObject(&imgSurface, &resourceDesc);

	dim3 grid = { unsigned((fractalData->size.w + 31) / 32), unsigned((fractalData->size.h + 31) / 32), 1 };
	int2 size = make_int2(fractalData->size.w, fractalData->size.h);
	float2 center = make_float2(fractalData->center[0], fractalData->center[1]);
	runMandelbrotFractal << < grid, { 32, 32, 1 }, 0, stream >> > (imgSurface, size, center, fractalData->scale, fractalData->iter);

	cudaDestroySurfaceObject(imgSurface);
	cudaGraphicsUnmapResources(1, &graphicsResources, stream);
	cudaStreamSynchronize(stream);
}

void MandelbrotFractalCUDA_Glue::close()
{
	//cudaGraphicsUnmapResources(1, &graphicsResources);
	cudaGraphicsUnregisterResource(graphicsResources);
}