#include <_ImGui.h>
#include <_Pair.h>

namespace Window
{
	ImGuiWindowManager* ImGuiWindowManager::__windowManager = nullptr;
	// ImGuiWindow
	ImGuiWindow::ImGuiWindow(Window::Data const& _data, bool _create_imgui_ctx, Window::CallbackFun const& _callback, GLFWcharfun _charFun, GuiBlock* _guiBlock)
		:
		window(_data, _callback),
		imguiContext(nullptr),
		imguiIO(nullptr),
		charFun(_charFun)
	{
		if (_create_imgui_ctx)
		{
			imguiContext = ImGui::CreateContext();
			if (imguiContext)
			{
				ImGui::SetCurrentContext(imguiContext);
				ImGui_ImplGlfw_InitForOpenGL(window.window, false);
				ImGui_ImplOpenGL3_Init("#version 460");
				imguiIO = &ImGui::GetIO();
			}
		}
		if (_guiBlock)
			guiBlocks.emplace_back(_guiBlock);
		glfwSetCharCallback(window.window, charFun);
	}

	ImGuiWindow::~ImGuiWindow()
	{
		if (window.window)
		{
			if (glfwWindowShouldClose(window.window))
			{
				makeCurrent();
				if (window.openGL)
					window.openGL->close();
				printf("Destroy window %p with imgui %p\n", window.window, imguiContext);
				if (imguiContext)
				{
					ImGui_ImplOpenGL3_Shutdown();
					ImGui_ImplGlfw_Shutdown();
					ImGui::DestroyContext(imguiContext);
					imguiContext = nullptr;
					imguiIO = nullptr;
				}
				glfwDestroyWindow(window.window);
			}
			window.window = nullptr;
		}
	}

	void ImGuiWindow::init(OpenGL::OpenGL* _openGL)
	{
		makeCurrent();
		window.init(_openGL);
	}

	void ImGuiWindow::makeCurrent()const
	{
		glfwMakeContextCurrent(window.window);
		if (imguiContext)
		{
			ImGui::SetCurrentContext(imguiContext);
		}
	}

	void ImGuiWindow::addGuiBlock(GuiBlock* _guiBlock)
	{
		guiBlocks.emplace_back(_guiBlock);
	}

	void ImGuiWindow::run()const
	{
		makeCurrent();
		if (window.openGL)
			window.openGL->run();
		if (imguiContext && guiBlocks.size())
		{
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();
			for (auto g : guiBlocks)
				if (g)g->gui();
			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		}
	}
}

namespace GUI
{
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

	Window::ImGuiWindow& UserInterface::createWindow(Window::Window::Data const& _data)
	{
		return wm.createWindow(_data);
	}

	void UserInterface::bindOpenGL(Window::ImGuiWindow& _window, OpenGL::OpenGL* _openGL)
	{
		_window.init(_openGL);
	}

	void UserInterface::bindOpenGLMain(OpenGL::OpenGL* _openGL)
	{
		wm.windows.begin->data.init(_openGL);
	}

	bool UserInterface::update()
	{
		// TODO: add multi thread support for rendering multiple window and their guis
		if (glfwWindowShouldClose(mainWindow->window.window))
		{
			wm.closeAllButFirst();
			wm.closeAll();
			return false;
		}
		wm.close();
		wm.render();
		glfwPollEvents();
		return true;
	}

	void UserInterface::minimalLoop()
	{
		while (update());
	}

	UserInterface* UserInterface::__userInterface = nullptr;
	OpenGL::OpenGLInit UserInterface::openglInit = OpenGL::OpenGLInit(4, 6);
}
