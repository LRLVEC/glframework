#include <_NBody.h>
#ifdef _CUDA
using NBodyImpl = OpenGL::NBodyCUDAImpl;
#else
using NBodyImpl = OpenGL::NBodyOpenGLImpl;
#endif
#include <_ImGui.h>

namespace GUI
{
	struct MultiViewGui :WindowGui
	{
		ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
		bool show_another_window = false;
		bool should_create_new_view = false;
		float f = 0.0f;
		int counter = 0;

		MultiViewGui(Window::Window* _window) :WindowGui(_window) {}
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

// there should be only one renderer!
// decouple the transform

struct MultiViewSim
{
	OpenGL::NBodyData nbodyData;
	NBodyImpl nbody;
	OpenGL::NBodyMultiViewRenderer renderer;
	MultiViewSim(unsigned int _blocks, bool _experiment, OpenGL::SourceManager* _sm)
		:
		nbodyData(_blocks, _experiment, _sm),
		nbody(&nbodyData, _sm),
		renderer(&nbody, _sm)
	{
	}
};

struct MultiView
{
	Window::Window::Data mainWindowData;
	GUI::UserInterface ui;
	OpenGL::SourceManager sm;
	MultiViewSim mainSim;
	GUI::MultiViewGui gui;

	MultiView()
		:
		mainWindowData{"Main View",{{1920, 1080}, /*resizable=*/true, /*fullscreen=*/false}},
		ui(mainWindowData),
		sm(String<char>("./")),
		mainSim(10 * 1, false, &sm),
		gui(nullptr)
	{
		mainSim.renderer.registerMainTransform(ui.mainWindow->window);
		// bind opengl before creating gui!
		ui.bindOpenGLMain(&mainSim.renderer);
		gui.create(ui.mainWindow);
		ui.registerWindowGui(&gui);
		::printf("Num particles: %d\n", mainSim.nbodyData.particles.particles.length);
	}

	void createNewView()
	{
		Window::Window::Data subWindowData{"New View", {{800, 800}, /*resizable=*/true, /*fullscreen=*/false}};
		Window::Window& w = ui.createWindow(subWindowData);
		mainSim.renderer.registerTransform(w.window);
		ui.bindOpenGL(w, &mainSim.renderer);
	}

	void loop()
	{
		while (ui.update())
		{
			if (gui.should_create_new_view)
			{
				gui.should_create_new_view = false;
				createNewView();
			}
		}
	}
};

int main()
{
	try
	{
		printf("Fuck!\n");
		MultiView test;
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