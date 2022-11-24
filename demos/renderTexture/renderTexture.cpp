#include <_TextureRenderer.h>
#include <_ImGui.h>
#include <_BMP.h>

namespace GUI
{
	struct TextureRendererGui :WindowGui
	{
		ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
		bool show_another_window = false;
		bool should_create_new_view = false;
		float f = 0.0f;
		int counter = 0;

		TextureRendererGui(Window::Window* _window) :WindowGui(_window) {}
		virtual void gui()override
		{
			//printf("gui()\n");
			makeCurrent();
			ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

			ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
			//ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
			ImGui::Checkbox("Another Window", &show_another_window);

			ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
			ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

			if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
				counter++;
			ImGui::SameLine();
			ImGui::Text("counter = %d", counter);

			if (ImGui::Button("New View"))
				should_create_new_view = true;

			ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
			ImGui::End();
			//bool show_demo_window = false;
			//ImGui::ShowDemoWindow(&show_demo_window);
			if (show_another_window)
			{
				ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
				ImGui::Text("Hello from another window!");
				if (ImGui::Button("Close Me"))
					show_another_window = false;
				ImGui::End();
			}
		}
	};
}

namespace OpenGL
{
	struct RenderToTexture: OpenGL
	{
		BMPData img;
		TextureRendererProgram renderer;
		
		RenderToTexture(SourceManager* _sm)
			:
			img("./resources/Frostbite.bmp"),
			renderer(_sm, { int(img.bmp.header.width), int(img.bmp.header.height) })
		{
			// this update method can be replaced by cuda functions or some off-screen rendering pipline.
			renderer.frameTexture.data = &img;
			renderer.frameConfig.dataInit(0, TextureInputRGB, TextureInputUByte);
		}
		virtual void init(FrameScale const& _size) override
		{
		}
		virtual void run() override
		{
			FrameScale frameSize;
			glfwGetWindowSize(glfwGetCurrentContext(), &frameSize.w, &frameSize.h);
			if (frameSize.w && frameSize.h)
			{
				glViewport(0, 0, frameSize.w, frameSize.h);
				glClearColor(1.0f, 1.0f, 0.0f, 0.0f);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				renderer.use();
				renderer.run();
			}
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
			}
		}
	};
}

struct RenderToTextureTest
{
	Window::Window::Data mainWindowData;
	GUI::UserInterface ui;
	OpenGL::SourceManager sm;
	OpenGL::RenderToTexture renderer;
	GUI::TextureRendererGui gui;

	RenderToTextureTest()
		:
		mainWindowData{"RenderTexture",{{720, 720}, /*resizable=*/true, /*fullscreen=*/false}},
		ui(mainWindowData),
		sm(String<char>("./")),
		renderer(&sm),
		gui(nullptr)
	{
		// bind opengl before creating gui!
		ui.bindOpenGLMain(&renderer);
		gui.create(ui.mainWindow);
		ui.registerWindowGui(&gui);
	}

	void loop()
	{
		while (ui.update())
		{
			// handle gui update
		}
	}
};

int main()
{
	try
	{
		printf("Fuck!\n");
		RenderToTextureTest test;
		//test.ui.minimalLoop();
		test.loop();
		return 0;
	}
	catch (const std::exception& e)
	{
		printf("%s", e.what());
		return 0;
	}
}