#include <_ImGui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_opengl3.h>
#include <_Pair.h>

namespace GUI
{
	// WindowGui
	WindowGui::WindowGui(Window::Window* _window)
		:
		window(nullptr),
		imguiContext(nullptr),
		imguiIO(nullptr)
	{
		create(_window);
	}

	WindowGui::~WindowGui()
	{
		destroy();
	}

	bool WindowGui::create(Window::Window* _window = nullptr)
	{
		if (_window || window)
		{
			// original window does not exist
			if (window == nullptr)
			{
				// new window does not exist
				if (_window == nullptr)
				{
					return false;
				}
				// new window does not have GLFWwindow
				else if (_window->window == nullptr)
				{
					return false;
				}
				window = _window;
			}
			// new window is different from original one and original window has GLFWwindow
			else if (_window != nullptr && window->window)
			{
				// new window has GLFWwindow and GLFWwindows are different
				if (_window->window && window->window != _window->window)
				{
					destroy();
					window = _window;
				}
			}
			if (window->window)
			{
				window->makeCurrent();
				imguiContext = ImGui::CreateContext();
				if (imguiContext)
				{
					ImGui::SetCurrentContext(imguiContext);
					ImGui_ImplGlfw_InitForOpenGL(window->window, true);
					ImGui_ImplOpenGL3_Init("#version 460");
					imguiIO = &ImGui::GetIO();
					return true;
				}
			}
		}
		return false;
	}

	void WindowGui::destroy()
	{
		if (imguiContext)
		{
			printf("Destroy Gui %p\n", imguiContext);
			makeCurrent();
			ImGui_ImplOpenGL3_Shutdown();
			ImGui_ImplGlfw_Shutdown();
			ImGui::DestroyContext(imguiContext);
			imguiContext = nullptr;
			imguiIO = nullptr;
		}
	}

	void WindowGui::makeCurrent()
	{
		if (imguiContext)
		{
			window->makeCurrent();
			ImGui::SetCurrentContext(imguiContext);
		}
	}

	void WindowGui::newFrame()
	{
		if (imguiContext)
		{
			makeCurrent();
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();
		}
	}

	void WindowGui::draw()
	{
		if (imguiContext)
		{
			makeCurrent();
			ImGui::Render();
		}
	}

	void WindowGui::render()
	{
		if (imguiContext)
		{
			makeCurrent();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		}
	}

	// UserInterface
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

	void UserInterface::bindOpenGL(Window::Window& _window, OpenGL::OpenGL* _openGL)
	{
		_window.init(_openGL);
	}

	void UserInterface::bindOpenGLMain(OpenGL::OpenGL* _openGL)
	{
		wm.makeCurrent(0);
		wm.init(0, _openGL);
	}

	void UserInterface::registerWindowGui(WindowGui* _windowGui)
	{
		if (_windowGui)
		{
			windowGuis.pushBack(_windowGui);
		}
	}

	void UserInterface::updateWindow(Window::Window& _window, Vector<WindowGui*>& _winidowGuis)
	{
		// already checked that this window is alive and pullEvents is done
		_window.makeCurrent();
		_winidowGuis.traverseLambda
		([](WindowGui* const& _windowGui)
			{
				_windowGui->newFrame();
				_windowGui->gui();
				_windowGui->draw();
			}
		);
		_window.openGL->run();
		([](WindowGui* const& _windowGui)
			{
				_windowGui->render();
			}
		);
		_window.swapBuffers();
	}

	bool UserInterface::update(int interval)
	{
		// TODO: add multi thread support for rendering multiple window and their guis
		glfwSwapInterval(interval);
		// destroy the guis that their windows were already destroyed
		windowGuis.check([](GUI::WindowGui* const& _windowGui)
			{
				if (glfwWindowShouldClose(_windowGui->window->window))
				{
					_windowGui->destroy();
					return false;
				}
				return true;
			});
		wm.windows.checkLambda([this](Window::Window const& _window)
			{
				if (glfwWindowShouldClose(_window.window))
				{
					glfwMakeContextCurrent(_window.window);
					glfwSetWindowShouldClose(_window.window, true);
					_window.openGL->close();
					//glfwDestroyWindow(_window.window);// move to deconstruction func
					return false;
				}
				return true;
			});
		if (wm.windows.length)
		{
			if (&wm.windows[0].data != mainWindow)
			{
				wm.closeAll();
				return false;
			}

			// only for main window and the first windowGui
			/*wm.pullEvents();
			windowGuis[0].data->newFrame();
			windowGuis[0].data->gui();
			windowGuis[0].data->draw();
			wm.render();
			windowGuis[0].data->render();
			wm.swapBuffers();*/

			// create lists of windowgui for each window
			Vector<Vector<WindowGui*>> windowGuiTable;
			for (int c0(0); c0 < wm.windows.length; ++c0)
			{
				windowGuiTable.pushBack(Vector<WindowGui*>());
			}
			windowGuis.traverseLambda([&windowGuiTable, this](WindowGui* const& _windowGui)
				{
					windowGuiTable[wm.windows.id(*_windowGui->window)].pushBack(_windowGui);
				}
			);

			// stuck!
			/*ThreadPool{}.parallelFor<int>(0, windowGuiTable.length, [&](int id)
				{
					updateWindow(wm.windows[id].data, windowGuiTable[id]);
				});*/
			int winNum(0);
			wm.windows.traverseLambda([&windowGuiTable, &winNum](Window::Window const& _window)
				{
					_window.makeCurrent();
					windowGuiTable[winNum].traverseLambda([](WindowGui* const& _windowGui)
						{
							_windowGui->newFrame();
							_windowGui->gui();
							_windowGui->draw();
						});
					if (_window.openGL)
						_window.openGL->run();
					windowGuiTable[winNum++].traverseLambda([](WindowGui* const& _windowGui)
						{
							_windowGui->render();
						});
				}
			);
			wm.swapBuffers();
			wm.pullEvents();
			return true;
		}
		return false;
	}

	void UserInterface::minimalLoop()
	{
		while (update(0));
	}

	UserInterface* UserInterface::__userInterface = nullptr;
	OpenGL::OpenGLInit UserInterface::openglInit = OpenGL::OpenGLInit(4, 6);
}
