#include <_Fractal.h>
#include <_ImGui.h>
#include <_BMP.h>

namespace GUI
{
	struct TextureRendererGui :WindowGui
	{
		OpenGL::MandelbrotFractalData* fractalData;
		ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
		bool show_another_window = false;
		bool should_create_new_view = false;
		float f = 0.0f;
		int counter = 0;


		TextureRendererGui(Window::Window* _window, OpenGL::MandelbrotFractalData* _fractalData)
			:
			WindowGui(_window),
			fractalData(_fractalData)
		{
		}
		virtual void gui()override
		{
			//printf("gui()\n");
			makeCurrent();
			ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

			ImGui::Text("Scale: %e", fractalData->scale);             // Display some text (you can use a format strings too)
			ImGui::Text("Center: %f, %f", fractalData->center[0], fractalData->center[1]);
			//ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
			//ImGui::Checkbox("Another Window", &show_another_window);

			//ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
			//ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

			//if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
			//	counter++;
			//ImGui::SameLine();
			//ImGui::Text("counter = %d", counter);

			//if (ImGui::Button("New View"))
			//	should_create_new_view = true;

			ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
			ImGui::End();
			//bool show_demo_window = false;
			//ImGui::ShowDemoWindow(&show_demo_window);
			//if (show_another_window)
			//{
			//	ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
			//	ImGui::Text("Hello from another window!");
			//	if (ImGui::Button("Close Me"))
			//		show_another_window = false;
			//	ImGui::End();
			//}
		}
	};
}

#ifdef _CUDA
struct MandelbrotFractal
{
	OpenGL::MandelbrotFractalData fractalData;
	OpenGL::MandelbrotFractalCUDAImpl fractalImpl;
	OpenGL::MandelbrotFractalRenderer renderer;
	MandelbrotFractal(OpenGL::FrameScale const& _size, OpenGL::SourceManager* _sm)
		:
		fractalData(_size),
		fractalImpl(&fractalData),
		renderer(&fractalImpl, _sm)
	{
	}
};
#else
#endif

struct TextureRendererTest
{
	Window::Window::Data mainWindowData;
	GUI::UserInterface ui;
	OpenGL::SourceManager sm;
	MandelbrotFractal fractal;
	GUI::TextureRendererGui gui;

	TextureRendererTest()
		:
		mainWindowData{ "RenderTexture",{{1920, 1080}, /*resizable=*/true, /*fullscreen=*/false} },
		ui(mainWindowData),
		sm(String<char>("./")),
		fractal(mainWindowData.size.size, &sm),
		gui(nullptr, &fractal.fractalData)
	{
		// bind opengl before creating gui!
		ui.bindOpenGLMain(&fractal.renderer);
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
		TextureRendererTest test;
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