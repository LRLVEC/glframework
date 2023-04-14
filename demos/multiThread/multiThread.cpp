#include <cstdio>
#include <_NBody.h>
#ifdef _CUDA
using NBodyImpl = OpenGL::NBodyCUDAImpl;
#else
using NBodyImpl = OpenGL::NBodyOpenGLImpl;
#endif
#include <_ImGui.h>

namespace GUI
{
	struct MultiSimGui :GuiBlock
	{
		ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
		bool show_another_window = false;
		bool should_create_new_sub_sim = false;
		bool vsync = false;
		bool refresh_vsync = true;
		float f = 0.0f;
		int counter = 0;


		virtual void gui()override
		{
			//printf("gui()\n");
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

			if (ImGui::Button("New SubSim"))
				should_create_new_sub_sim = true;

			if (ImGui::Button("V-sync (disable to get higher fps)"))
			{
				vsync = !vsync;
				refresh_vsync = true;
			}

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

struct SingleSim
{
	OpenGL::NBodyData nbodyData;
	NBodyImpl nbody;
	OpenGL::NBodySingleViewRenderer renderer;
	SingleSim(unsigned int _blocks, bool _experiment, OpenGL::SourceManager* _sm)
		:
		nbodyData(_blocks, _experiment, _sm),
		nbody(&nbodyData, _sm),
		renderer(&nbody, _sm)
	{
	}
};

struct MultiSimTest
{
	Window::Window::Data mainWindowData;
	GUI::UserInterface ui;
	OpenGL::SourceManager sm;
	SingleSim mainSim;
	GUI::MultiSimGui gui;
	List<Pair<Window::ImGuiWindow*, SingleSim*>> subSims;

	MultiSimTest()
		:
		mainWindowData{"MainSim",{{1920, 1080}, /*resizable=*/true, /*fullscreen=*/false}},
		ui(mainWindowData),
		sm(String<char>("./")),
		mainSim(10 * 1, false, &sm)
	{
		ui.bindOpenGLMain(&mainSim.renderer);
		ui.mainWindow->addGuiBlock(&gui);

		::printf("Num particles: %d\n", mainSim.nbodyData.particles.particles.length);
	}

	Pair<Window::ImGuiWindow*, SingleSim*> createSubSim()
	{
		Window::Window::Data subWindowData{"SubSim", {{800, 800}, /*resizable=*/true, /*fullscreen=*/false}};
		Window::ImGuiWindow& w = ui.createWindow(subWindowData);
		SingleSim* subSim(new SingleSim(10 * 1, false, &sm));
		::printf("Num particles: %d\n", subSim->nbodyData.particles.particles.length);
		ui.bindOpenGL(w, &subSim->renderer);
		return Pair<Window::ImGuiWindow*, SingleSim*>(&w, subSim);
	}

	void deleteUnusedSubSims()
	{
		subSims.checkLambda
		([this](Pair<Window::ImGuiWindow*, SingleSim*>const& _pair)
			{
				if (!ui.wm.exists(_pair.data0))
				{
					delete _pair.data1;
					return false;
				}
				return true;
			}
		);
	}

	void loop()
	{
		while (ui.update())
		{
			if (gui.refresh_vsync)
			{
				ui.wm.swapInterval(gui.vsync);
				gui.refresh_vsync = false;
			}
			deleteUnusedSubSims();
			if (gui.should_create_new_sub_sim)
			{
				gui.should_create_new_sub_sim = false;
				subSims.pushBack(createSubSim());
				ui.wm.swapInterval(gui.vsync);
			}
		}
	}
};

int main()
{
	try
	{
		printf("Fuck!\n");
		MultiSimTest test;
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