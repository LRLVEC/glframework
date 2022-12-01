#pragma once
#include <_TextureRenderer.h>

struct MandelbrotFractalCUDA_Glue;

namespace OpenGL
{
	struct MandelbrotFractalData
	{
		FrameScale size;
		Math::vec2<double> center;
		double scale;
		int iter;
		bool useDouble;
		bool useDoublePre;

		MandelbrotFractalData(FrameScale const& _size)
			:
			size(_size),
			center{ -0.5, 0. },
			scale(0.3),
			iter(2000),
			useDouble(false),
			useDoublePre(false)
		{
		}
		void update(Transform2D const& _trans)
		{
			size = _trans.size;
			center = _trans.center;
			scale = _trans.scale;
		}
	};

	struct MandelbrotFractalImplBase
	{
		MandelbrotFractalData* fractalData;
		MandelbrotFractalImplBase(MandelbrotFractalData* _mandelbrotFractalData)
			:
			fractalData(_mandelbrotFractalData)
		{
		}
		virtual void resize(TextureConfig<TextureStorage2D>* _texture) = 0;
		virtual void run() = 0;
		virtual void close() = 0;
	};

	struct MandelbrotFractalRenderer : OpenGL
	{
		MandelbrotFractalImplBase* fractal;
		TextureRendererProgram renderer;
		Transform2D trans;

		MandelbrotFractalRenderer(MandelbrotFractalImplBase* _fractal, SourceManager* _sm)
			:
			fractal(_fractal),
			renderer(_sm, _fractal->fractalData->size),
			trans({ _fractal->fractalData->size, {0.8,0.8,0.1,0.01}, {0.01}, 1.0 })
		{
		}
		virtual void init(FrameScale const& _size) override
		{
			renderer.resizeTexture(_size);
			trans.resize(_size.w, _size.h);
			fractal->fractalData->update(trans);
			// register cuda resources etc.
			fractal->resize(&renderer.frameConfig);
		}
		virtual void run() override
		{
			if (glfwWindowShouldClose(glfwGetCurrentContext()))
			{
				return;
			}
			FrameScale frameSize;
			glfwGetWindowSize(glfwGetCurrentContext(), &frameSize.w, &frameSize.h);
			if (frameSize.w && frameSize.h)
			{
				if (frameSize != trans.size)
				{
					renderer.resizeTexture(frameSize);
					trans.resize(frameSize.w, frameSize.h);
					fractal->resize(&renderer.frameConfig);
				}
				trans.operate();
				if (trans.updated || fractal->fractalData->useDouble != fractal->fractalData->useDoublePre)
				{
					trans.updated = false;
					fractal->fractalData->update(trans);
					fractal->fractalData->useDoublePre = fractal->fractalData->useDouble;
					fractal->run();
				}
				glViewport(0, 0, frameSize.w, frameSize.h);
				glClearColor(1.0f, 1.0f, 0.0f, 0.0f);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				renderer.use();
				renderer.run();
			}
		}
		virtual void frameSize(GLFWwindow* _window, int _w, int _h) override
		{
			if (glfwWindowShouldClose(_window))
			{
				return;
			}
			renderer.resizeTexture(FrameScale{ _w, _h });
			// register cuda resources etc.
			fractal->resize(&renderer.frameConfig);
			trans.resize(_w, _h);
		}
		virtual void mouseButton(GLFWwindow* _window, int _button, int _action, int _mods) override
		{
			if (glfwWindowShouldClose(_window))
			{
				return;
			}
			switch (_button)
			{
			case GLFW_MOUSE_BUTTON_LEFT:trans.mouse.refreshButton(0, _action); break;
			case GLFW_MOUSE_BUTTON_MIDDLE:trans.mouse.refreshButton(1, _action); break;
			case GLFW_MOUSE_BUTTON_RIGHT:trans.mouse.refreshButton(2, _action); break;
			}
		}
		virtual void mousePos(GLFWwindow* _window, double _x, double _y) override
		{
			if (glfwWindowShouldClose(_window))
			{
				return;
			}
			trans.mouse.refreshPos(_x, _y);
		}
		virtual void mouseScroll(GLFWwindow* _window, double _x, double _y) override
		{
			if (glfwWindowShouldClose(_window))
			{
				return;
			}
			if (_y != 0.0)
				trans.scroll.refresh(_y);
		}
		virtual void key(GLFWwindow* _window, int _key, int _scancode, int _action, int _mods) override
		{
			if (glfwWindowShouldClose(_window))
			{
				return;
			}
			switch (_key)
			{
			case GLFW_KEY_ESCAPE:
				if (_action == GLFW_PRESS)
				{
					glfwSetWindowShouldClose(_window, true);
				}
				break;
			case GLFW_KEY_A:trans.key.refresh(0, _action); break;
			case GLFW_KEY_D:trans.key.refresh(1, _action); break;
			case GLFW_KEY_W:trans.key.refresh(2, _action); break;
			case GLFW_KEY_S:trans.key.refresh(3, _action); break;
			}
		}
	};

#ifdef _CUDA
	struct MandelbrotFractalCUDAImpl : MandelbrotFractalImplBase
	{
		MandelbrotFractalCUDA_Glue* glue;

		MandelbrotFractalCUDAImpl(MandelbrotFractalData* _fractalData);
		~MandelbrotFractalCUDAImpl();

		virtual void resize(TextureConfig<TextureStorage2D>* _texture)override;
		virtual void run()override;
		virtual void close() override;
	};
#endif
}