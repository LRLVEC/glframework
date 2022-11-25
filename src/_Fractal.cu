#include <_Fractal.h>
#include <CUDA/_CUDA_Fractal.h>


namespace OpenGL
{
	MandelbrotFractalCUDAImpl::MandelbrotFractalCUDAImpl(MandelbrotFractalData* _fractalData)
		:
		MandelbrotFractalImplBase(_fractalData)
	{
		glue = new MandelbrotFractalCUDA_Glue(_fractalData);
	}

	MandelbrotFractalCUDAImpl::~MandelbrotFractalCUDAImpl()
	{
		close();
	}

	void MandelbrotFractalCUDAImpl::resize(TextureConfig<TextureStorage2D>* _texture)
	{
		glue->resize(_texture->texture->texture);
	}

	void MandelbrotFractalCUDAImpl::run()
	{
		glue->run();
	}

	void MandelbrotFractalCUDAImpl::close()
	{
		delete glue;
		glue = nullptr;
	}
}