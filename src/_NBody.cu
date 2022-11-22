#include <CUDA/_CUDA.h>
#include <CUDA/_CUDA_NBody_Common.h>
#include <_NBody.h>

namespace OpenGL
{
	NBodyCUDAImpl::NBodyCUDAImpl(NBodyData* _nbodyData, SourceManager* _sm)
		:
		NBodyImplBase(_nbodyData)
	{
		particlesBufferCUDA = new CUDA::Buffer(CUDA::Buffer::GLinterop);
		glue = new NBodyCUDA_Glue(nbodyData->particles.num / 1024, 0.005f, 0.001f);
	}

	NBodyCUDAImpl::~NBodyCUDAImpl()
	{
		delete particlesBufferCUDA;
		delete glue;
	}

	void NBodyCUDAImpl::experiment()
	{
		CUDA::Buffer expBuffer(CUDA::Buffer::Device);
		expBuffer.resize(nbodyData->particles.particles.length * sizeof(ExpData));
		glue->experiment((ExpData*)expBuffer.device);
		expBuffer.moveToHost();
		File file("./");
		String<char> answer;
		for (int c0(0); c0 < nbodyData->particles.particles.length; ++c0)
		{
			char tp[50];
			ExpData& data(((ExpData*)expBuffer.host)[c0]);
			sprintf(tp, "%f %f\n", data.r, data.force);
			answer += tp;
		}
		file.createText("answer.txt", answer);
	}

	void NBodyCUDAImpl::init()
	{
		particlesBufferCUDA->resize(nbodyData->particlesArray.buffer->buffer);
		glue->particles = (NBodyCUDAParticle*)particlesBufferCUDA->map();
	}

	void NBodyCUDAImpl::run()
	{
		glFinish();
		glue->run();
	}

	void NBodyCUDAImpl::close()
	{
		particlesBufferCUDA->unmap();
	}
}