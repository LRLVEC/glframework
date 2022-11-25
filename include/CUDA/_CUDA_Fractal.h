#pragma once
#include <_OpenGL.h>
#include <_Fractal.h>
#include <cuda_runtime.h>

struct MandelbrotFractalCUDA_Glue
{
	cudaStream_t stream;
	OpenGL::MandelbrotFractalData* fractalData;
	cudaGraphicsResource_t graphicsResources;
	cudaArray_t imgArray; // read only
	cudaSurfaceObject_t imgSurface; // read and write

	MandelbrotFractalCUDA_Glue(OpenGL::MandelbrotFractalData* _fractalData, GLuint _texture = 0);
	~MandelbrotFractalCUDA_Glue();
	// after TextureConfig<TextureStorage2D>::resize()
	void resize(GLuint _texture);
	void run();
	void close();
};