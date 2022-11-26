#pragma once
#include <CUDA/_CUDA.h>
#include <_Fractal.h>
#include <cuda_runtime.h>

struct MandelbrotFractalCUDA_Glue
{
	cudaStream_t stream;
	OpenGL::MandelbrotFractalData* fractalData;
	CUDA::GLTexture<2> img;
	//cudaGraphicsResource_t graphicsResources;
	//cudaArray_t imgArray; // read only
	//cudaSurfaceObject_t imgSurface; // read and write

	MandelbrotFractalCUDA_Glue(OpenGL::MandelbrotFractalData* _fractalData, OpenGL::TextureConfig<OpenGL::TextureStorage2D>* _textureConfig = nullptr);
	~MandelbrotFractalCUDA_Glue();
	// after TextureConfig<TextureStorage2D>::resize()
	void resize(OpenGL::TextureConfig<OpenGL::TextureStorage2D>* _textureConfig);
	void run();
	void close();
};