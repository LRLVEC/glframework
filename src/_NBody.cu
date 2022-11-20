#include <CUDA/_CUDA.h>
#include <CUDA/_CUDA_NBody_Common.h>
#include <_NBody.h>

namespace OpenGL
{
	NBodyCUDA::NBodyCUDA(unsigned int _blocks, bool _experiment, String<char>const& _smPath)
		:
		sm(_smPath),
		particles(_blocks * 1024),
		particlesData(&particles),
		particlesBuffer(&particlesData),
		trans({ {80.0,0.1,800},{0.8,0.8,0.1},{1},500.0 }),
		renderer(&sm, &particlesBuffer, &trans)
	{
		particlesBufferCUDA = new CUDA::Buffer(CUDA::Buffer::GLinterop);
		glue = new NBodyCUDA_Glue(_blocks, 0.005f, 0.001f);
		if (_experiment)
			particles.experimentGalaxy();
		else
			particles.randomGalaxyOptimized();
	}

	NBodyCUDA::~NBodyCUDA()
	{
		delete particlesBufferCUDA;
		delete glue;
	}

	void NBodyCUDA::experiment()
	{
		CUDA::Buffer expBuffer(CUDA::Buffer::Device);
		expBuffer.resize(particles.particles.length * sizeof(ExpData));
		glue->experiment((ExpData*)expBuffer.device);
		expBuffer.moveToHost();
		File file("./");
		String<char> answer;
		for (int c0(0); c0 < particles.particles.length; ++c0)
		{
			char tp[50];
			ExpData& data(((ExpData*)expBuffer.host)[c0]);
			sprintf(tp, "%f %f\n", data.r, data.force);
			answer += tp;
		}
		file.createText("answer.txt", answer);
	}


	void NBodyCUDA::init(FrameScale const& _size)
	{
		glViewport(0, 0, _size.w, _size.h);
		glPointSize(2);
		glEnable(GL_DEPTH_TEST);
		trans.init(_size);
		renderer.transUniform.dataInit();
		renderer.particlesArray.dataInit();
		particlesBufferCUDA->resize(renderer.particlesArray.buffer->buffer);
		glue->particles = (NBodyCUDAParticle*)particlesBufferCUDA->map();
	}

	void NBodyCUDA::run()
	{
		trans.operate();
		if (trans.updated)
		{
			renderer.transUniform.refreshData();
			trans.updated = false;
		}
		renderer.use();
		renderer.run();
		glFinish();
		glue->run();
	}

	
	void NBodyCUDA::key(GLFWwindow* _window, int _key, int _scancode, int _action, int _mods)
	{
		switch (_key)
		{
		case GLFW_KEY_ESCAPE:
			if (_action == GLFW_PRESS)
			{
				particlesBufferCUDA->unmap();
				glfwSetWindowShouldClose(_window, true);
			}
			break;
		case GLFW_KEY_A:trans.key.refresh(0, _action); break;
		case GLFW_KEY_D:trans.key.refresh(1, _action); break;
		case GLFW_KEY_W:trans.key.refresh(2, _action); break;
		case GLFW_KEY_S:trans.key.refresh(3, _action); break;
		}
	}
}