#include <CUDA/_CUDA_Fractal.h>
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_gl_interop.h>
#include <CUDA/helper_math.h>

// detailed impl of glue

// float version
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

// double version
__device__ double mandelbrotFractalKernel(double2 c, int iter)
{
	double c2 = dot(c, c);
	// skip computation inside M1 - https://iquilezles.org/articles/mset1bulb
	if (256. * c2 * c2 - 96. * c2 + 32. * c.x - 3. < 0.) return 0.;
	// skip computation inside M2 - https://iquilezles.org/articles/mset2bulb
	if (16. * (c2 + 2. * c.x + 1.) - 1. < 0.) return 0.;

	int B = iter / 2;//256.f;
	double l = 0.;
	double2 z = make_double2(0.);
	for (int i = 0; i < iter; i++)
	{
		z = make_double2(z.x * z.x - z.y * z.y, 2. * z.x * z.y) + c;
		if (dot(z, z) > (B * B)) break;
		l += 1.;
	}
	if (l > iter - 1) return 0.;

	double sl = l - std::log(std::log(dot(z, z))) + 4.;
	return sl;
}

// 32 * 32 block
__global__ void runMandelbrotFractal(cudaSurfaceObject_t img, int2 _size, double2 _center, double scale, int iter, bool useDouble = false)
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
				if (useDouble)
				{
					double2 p = _center + make_double2(x - _size.x / 2 + m / double(AA), _size.y / 2 - y + n / double(AA)) / (double(_size.x) * scale);
					double l = 3. + mandelbrotFractalKernel(p, iter) * 0.15;
					// col += 0.5f * (1.f + make_float3(std::cos(l + 0.f), std::cos(l + 0.6f), std::cos(l + 1.f)));
					col += make_float3(0.5 * (1. + make_double3(cos(l + 0.), cos(l + 0.6), cos(l + 1.))));
				}
				else
				{
					float2 p = make_float2(_center) + make_float2(x - _size.x / 2 + m / float(AA), _size.y / 2 - y + n / float(AA)) / (float(_size.x) * scale);
					float l = 3.f + mandelbrotFractalKernel(p, iter) * 0.15f;
					col += 0.5f * (1.f + make_float3(__cosf(l + 0.f), __cosf(l + 0.6f), __cosf(l + 1.f)));
				}
			}
		col /= float(AA * AA);
		surf2Dwrite(make_float4(col, 1.f), img, x * sizeof(float4), y);
	}
}

MandelbrotFractalCUDA_Glue::MandelbrotFractalCUDA_Glue(OpenGL::MandelbrotFractalData* _fractalData, OpenGL::TextureConfig<OpenGL::TextureStorage2D>* _textureConfig)
	:
	fractalData(_fractalData)
{
	cudaStreamCreate(&stream);
	if (_textureConfig)
	{
		resize(_textureConfig);
	}
}

MandelbrotFractalCUDA_Glue::~MandelbrotFractalCUDA_Glue()
{
	close();
	cudaStreamDestroy(stream);
}

void MandelbrotFractalCUDA_Glue::resize(OpenGL::TextureConfig<OpenGL::TextureStorage2D>* _textureConfig)
{
	img.registerImage(*_textureConfig, cudaGraphicsRegisterFlagsSurfaceLoadStore);
}

void MandelbrotFractalCUDA_Glue::run()
{
	cudaStreamSynchronize(stream);
	img.map(stream);
	img.createArray(0);
	img.createSurface();

	dim3 grid = { unsigned((fractalData->size.w + 31) / 32), unsigned((fractalData->size.h + 31) / 32), 1 };
	int2 size = make_int2(fractalData->size.w, fractalData->size.h);
	double2 center = make_double2(fractalData->center[0], fractalData->center[1]);
	runMandelbrotFractal << < grid, { 32, 32, 1 }, 0, stream >> > (img.surface, size, center, fractalData->scale, fractalData->iter, fractalData->useDouble);

	img.destroySurface();
	img.unmap(stream);
	cudaStreamSynchronize(stream);
}

void MandelbrotFractalCUDA_Glue::close()
{
	img.unregisterResource();
}