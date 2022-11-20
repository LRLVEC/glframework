#include <_ImGui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_opengl3.h>


namespace GUI
{
	UserInterface::UserInterface(Window::Window::Data const& _data)
		:
		wm(_data),
		mainWindow(&wm.windows.begin->data)
	{
		if (__userInterface)throw std::runtime_error{ "Cannot construct new UserInterface if there exists one!" };
		__userInterface = this;
		printInfo();
	}

	void UserInterface::printInfo()
	{
		openglInit.printRenderer();
	}

	Window::Window& UserInterface::createWindow(Window::Window::Data const& _data)
	{
		wm.createWindow(_data);
		return wm.windows.end->data;
	}

	void UserInterface::bindWindow(Window::Window& _window, OpenGL::OpenGL* _openGL)
	{
		_window.init(_openGL);
	}

	void UserInterface::bindMainWindow(OpenGL::OpenGL* _openGL)
	{
		wm.makeCurrent(0);
		wm.init(0, _openGL);
	}

	void UserInterface::mainLoop()
	{
		glfwSwapInterval(0);
		while (!wm.close())
		{
			if (&wm.windows[0].data != mainWindow)
			{
				wm.closeAll();
			}
			wm.pullEvents();
			
			wm.render();
			wm.swapBuffers();
		}
	}

	UserInterface* UserInterface::__userInterface = nullptr;
	OpenGL::OpenGLInit UserInterface::openglInit = OpenGL::OpenGLInit(4, 6);

	void ImGuiBase::init()
	{
		ImGui::CreateContext();
	}
}
