#pragma once
#include <_OpenGL.h>
#include <_Window.h>
#include <_ThreadPool.h>
#include <imgui/imgui.h>

namespace GUI
{
	// imgui config base for a single window
	struct WindowGui
	{
		Window::Window* window;
		ImGuiContext* imguiContext;
		ImGuiIO* imguiIO;

		WindowGui(Window::Window* _window = nullptr);
		~WindowGui();
		// create imgui context and io
		bool create(Window::Window* _window);
		// destroy imgui context and io, remain window to create again
		void destroy();
		// make imguiContext current
		void makeCurrent();
		// imgui new frame, call after glfwPullEvents()
		void newFrame();
		// generate imgui draw data
		void draw();
		// render draw data
		void render();
		// user defined gui function, draw imgui windows etc.
		virtual void gui() = 0;
	};

	// multi-window user interface, have a main window and any number of sub windows
	struct UserInterface
	{
		// only one UserInterface instance is allowed
		static UserInterface* __userInterface;
		static OpenGL::OpenGLInit openglInit;
		Window::WindowManager wm;
		Window::Window* mainWindow;
		List<WindowGui*> windowGuis;

		UserInterface() = delete;
		UserInterface(Window::Window::Data const& _data);
		// print render gpu infomation
		static void printInfo();
		// create new window
		Window::Window& createWindow(Window::Window::Data const& _data);

		// bind OpenGL class to window
		void bindOpenGL(Window::Window& _window, OpenGL::OpenGL* _openGL);

		// bind OpenGL class to main window
		void bindOpenGLMain(OpenGL::OpenGL* _openGL);

		// register WindowGui, life time is controlled by user like openGL, one window may have multiple WindowGui
		void registerWindowGui(WindowGui* _windowGui);

		// cannot work!!!!!!!!!!!!!!!
		// single window render pipline for a thread, includeing its OpenGL func and WindowGui, multiple WindowGui will be supported soon
		void updateWindow(Window::Window& _window, Vector<WindowGui*>& _winidowGuis);

		// get inputs and update one frame for all windows, returns whether main window is alive, _interval is for glfwSwapInterval
		bool update(int _interval);

		// minimal loop of drawing windows and their UIs. Since multi-context imgui is not supported yet (comming soon), 
		// so use 1 WindowGui at most.
		void minimalLoop();
	};
}