#pragma once
#include <_OpenGL.h>
#include <_Window.h>
#include <imgui/imgui.h>

namespace GUI
{
	// multi-window user interface, have a main window and any number of sub window
	struct UserInterface
	{
		// only one UserInterface instance is allowed
		static UserInterface* __userInterface;
		static OpenGL::OpenGLInit openglInit;
		Window::WindowManager wm;
		Window::Window* mainWindow;

		UserInterface() = delete;
		UserInterface(Window::Window::Data const& _data);
		// print render gpu infomation
		static void printInfo();
		// create new window
		Window::Window& createWindow(Window::Window::Data const& _data);
		// bind OpenGL class to window
		void bindWindow(Window::Window& _window, OpenGL::OpenGL* _openGL);
		// bind OpenGL class to main window
		void bindMainWindow(OpenGL::OpenGL* _openGL);
		// draw window and ui main loop
		void mainLoop();
	};
	
	// imgui config for a single window
	struct ImGuiBase
	{
		static void init();
	};
}