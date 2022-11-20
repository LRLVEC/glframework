#include <cstdio>
#include <_NBody.h>
#ifdef _CUDA
using NBody = OpenGL::NBodyCUDA;
#else
using NBody = OpenGL::NBody;
#endif
#include <_ImGui.h>

namespace GUI
{
	struct TestGui :WindowGui
	{
		ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
		bool show_another_window = false;
		bool should_create_new_sub_sim = false;
		float f = 0.0f;
		int counter = 0;

		TestGui(Window::Window* _window) :WindowGui(_window) {}
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

			if (ImGui::Button("New SubSim"))
				should_create_new_sub_sim = true;

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

struct Test
{
	Window::Window::Data mainWindowData;
	GUI::UserInterface ui;
	NBody mainSim;
	GUI::TestGui gui;
	List<Pair<Window::Window*, NBody*>> subSims;

	Test()
		:
		mainWindowData{"MainSim",{{1920, 1080}, /*resizable=*/true, /*fullscreen=*/false}},
		ui(mainWindowData),
		mainSim(10 * 1, false, String<char>("./")),
		gui(nullptr)
	{
		// bind opengl before creating gui!
		ui.bindOpenGLMain(&mainSim);
		gui.create(ui.mainWindow);
		ui.registerWindowGui(&gui);

		::printf("Num particles: %d\n", mainSim.particles.particles.length);
	}

	Pair<Window::Window*, NBody*> createSubSim()
	{
		Window::Window::Data subWindowData{"SubSim", {{800, 800}, /*resizable=*/true, /*fullscreen=*/false}};
		Window::Window& w = ui.createWindow(subWindowData);
		NBody* subSim(new NBody(10 * 1, false, String<char>("./")));
		::printf("Num particles: %d\n", subSim->particles.particles.length);
		ui.bindOpenGL(w, subSim);
		return Pair<Window::Window*, NBody*>(&w, subSim);
	}

	void deleteUnusedSubSims()
	{
		subSims.checkLambda
		([this](Pair<Window::Window*, NBody*>const& _pair)
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
			deleteUnusedSubSims();
			if (gui.should_create_new_sub_sim)
			{
				gui.should_create_new_sub_sim = false;
				subSims.pushBack(createSubSim());
			}
		}
	}
};

int main()
{
	try
	{
		printf("Fuck!\n");
		Test test;
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