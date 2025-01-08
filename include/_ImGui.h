#pragma once
#include <_OpenGL.h>
#include <_Window.h>
#include <_ThreadPool.h>
#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_opengl3.h>

namespace GUI
{
	struct GuiBlock
	{
		virtual void gui() = 0;
		virtual ~GuiBlock() {}
	};
}

namespace Window
{
	using GuiBlock = GUI::GuiBlock;
	// window with imgui context
	struct WindowWithImGui
	{
		Window window;
		ImGuiContext* imguiContext;
		ImGuiIO* imguiIO;
		GLFWcharfun charFun;// for imgui
		std::vector<GuiBlock*> guiBlocks;

		WindowWithImGui(Window::Data const& _data, bool _create_imgui_ctx, Window::CallbackFun const& _callback, GLFWcharfun _charFun, GuiBlock* _guiBlock);
		~WindowWithImGui();
		bool operator==(WindowWithImGui const& _window) const
		{
			return window.window == _window.window.window;
		}
		bool operator==(GLFWwindow* const _window) const
		{
			return window.window == _window;
		}
		void init(OpenGL::OpenGL* _openGL);
		void makeCurrent()const;
		void addGuiBlock(GuiBlock* _guiBlock);
		void run()const;
	};

	// Imgui Window Manager
	struct ImGuiWindowManager
	{
		static ImGuiWindowManager* __windowManager;
		static bool imguiWantCaptureMouse();
		static bool imguiWantCaptureKeyboard();
		static void frameSizeCallback(GLFWwindow* _window, int _w, int _h)
		{
			WindowWithImGui& w = __windowManager->find(_window);
			w.makeCurrent();
			w.window.openGL->frameSize(_window, _w, _h);
		}
		static void framePosCallback(GLFWwindow* _window, int _w, int _h)
		{
			WindowWithImGui& w = __windowManager->find(_window);
			w.makeCurrent();
			w.window.openGL->framePos(_window, _w, _h);
		}
		static void frameFocusCallback(GLFWwindow* _window, int _focused)
		{
			WindowWithImGui& w = __windowManager->find(_window);
			w.makeCurrent();
			w.window.openGL->frameFocus(_window, _focused);
			if(w.imguiContext)
				ImGui_ImplGlfw_WindowFocusCallback(w.window.window, _focused);
		}
		static void mouseButtonCallback(GLFWwindow* _window, int _button, int _action, int _mods)
		{
			WindowWithImGui& w = __windowManager->find(_window);
			w.makeCurrent();
			if (!imguiWantCaptureMouse())
				w.window.openGL->mouseButton(_window, _button, _action, _mods);
			if(w.imguiContext)
				ImGui_ImplGlfw_MouseButtonCallback(w.window.window, _button, _action, _mods);
		}
		static void mousePosCallback(GLFWwindow* _window, double _x, double _y)
		{
			WindowWithImGui& w = __windowManager->find(_window);
			w.makeCurrent();
			if (!imguiWantCaptureMouse())
				w.window.openGL->mousePos(_window, _x, _y);
			if(w.imguiContext)
				ImGui_ImplGlfw_CursorPosCallback(w.window.window, _x, _y);
		}
		static void mouseScrollCallback(GLFWwindow* _window, double _x, double _y)
		{
			WindowWithImGui& w = __windowManager->find(_window);
			w.makeCurrent();
			if (!imguiWantCaptureMouse())
				w.window.openGL->mouseScroll(_window, _x, _y);
			if(w.imguiContext)
				ImGui_ImplGlfw_ScrollCallback(w.window.window, _x, _y);
		}
		static void keyCallback(GLFWwindow* _window, int _key, int _scancode, int _action, int _mods)
		{
			WindowWithImGui& w = __windowManager->find(_window);
			w.makeCurrent();
			if (!imguiWantCaptureKeyboard())
				w.window.openGL->key(_window, _key, _scancode, _action, _mods);
			if(w.imguiContext)
				ImGui_ImplGlfw_KeyCallback(w.window.window, _key, _scancode, _action, _mods);
		}
		static void charCallback(GLFWwindow* _window, unsigned int _c)
		{
			WindowWithImGui& w = __windowManager->find(_window);
			w.makeCurrent();
			if(w.imguiContext)
				ImGui_ImplGlfw_CharCallback(w.window.window, _c);
		}
		static constexpr Window::CallbackFun callbackFun
		{
			{
				frameSizeCallback,
					framePosCallback,
					frameFocusCallback
			},
			{
				mouseButtonCallback,
				mousePosCallback,
				mouseScrollCallback,
				keyCallback
			}
		};

		List<WindowWithImGui> windows;

		ImGuiWindowManager()
		{
			if (__windowManager)throw std::runtime_error{ "Cannot construct new ImGuiWindowManager if there exists one!" };
			__windowManager = this;
		}
		ImGuiWindowManager(Window::Data const& _data, GuiBlock* _guiBlock = nullptr, bool _create_imgui_ctx = true): ImGuiWindowManager()
		{
			windows.pushBack(WindowWithImGui(_data, _create_imgui_ctx, callbackFun, charCallback, _guiBlock));
		}
		WindowWithImGui& createWindow(Window::Data const& _data, GuiBlock* _guiBlock = nullptr, bool _create_imgui_ctx = true)
		{
			windows.pushBack(WindowWithImGui(_data, _create_imgui_ctx, callbackFun, charCallback, _guiBlock));
			return windows.end->data;
		}
		void makeCurrent(unsigned int _num)
		{
			windows[_num].data.makeCurrent();
		}
		WindowWithImGui& find(GLFWwindow* const _window)
		{
			return windows.find(_window).data;
		}
		bool exists(WindowWithImGui* _window)const
		{
			return windows.id(*_window) >= 0;
		}
		void swapInterval(uint32_t _interval)
		{
			windows.traverseLambda
			([this, _interval](WindowWithImGui const& _window)
				{
					_window.makeCurrent();
					// only set the main window _interval
					glfwSwapInterval(windows.begin->data == _window ? _interval : 0);
					return true;
				}
			);
		}
		void render()
		{
			windows.traverse
			([](WindowWithImGui const& _window)
				{
					_window.run();
					_window.window.swapBuffers();
					return true;
				}
			);
		}
		bool close()
		{
			windows.check([](WindowWithImGui const& _window)
				{
					if (glfwWindowShouldClose(_window.window.window))
						return false;
					return true;
				});
			if (!windows.length)return true;
			return false;
		}
		void closeAll()
		{
			windows.check([](WindowWithImGui const& _window)
				{
					_window.makeCurrent();
					glfwSetWindowShouldClose(_window.window.window, true);
					_window.window.openGL->close();
					return false;
				}
			);
		}
		void closeAllButFirst()
		{
			windows.checkLambda([this](WindowWithImGui const& _window)
				{
					if (_window == windows.begin->data)
						return true;
					_window.makeCurrent();
					glfwSetWindowShouldClose(_window.window.window, true);
					_window.window.openGL->close();
					return false;
				}
			);
		}
	};
}

namespace GUI
{
	// multi-window user interface, have a main window and any number of sub windows
	struct UserInterface
	{
		// only one UserInterface instance is allowed
		static UserInterface* __userInterface;
		static OpenGL::OpenGLInit openglInit;
		Window::ImGuiWindowManager wm;
		// Window::WindowManager wm;
		Window::WindowWithImGui* mainWindow;

		UserInterface() = delete;
		UserInterface(Window::Window::Data const& _data, bool _create_imgui_ctx = true);
		// print render gpu infomation
		static void printInfo();
		// create new window
		Window::WindowWithImGui& createWindow(Window::Window::Data const& _data);

		// bind OpenGL class to window
		void bindOpenGL(Window::WindowWithImGui& _window, OpenGL::OpenGL* _openGL);

		// bind OpenGL class to main window
		void bindOpenGLMain(OpenGL::OpenGL* _openGL);

		// whether the main window should be destroyed
		bool shouldExit();

		// exit
		void exit();

		// get inputs and update one frame for all windows, returns whether main window is alive, _interval is for glfwSwapInterval
		bool update();

		// minimal loop of drawing windows and their UIs. Since multi-context imgui is not supported yet (coming soon), 
		// so use 1 WindowGui at most.
		void minimalLoop();
	};
}