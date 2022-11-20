#include <cstdio>
#include <_Window.h>
#include <_NBody.h>
#include <_Math.h>
#include <_ImGui.h>

namespace GUI
{
	struct TestGui :WindowGui
	{
		ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
		bool show_another_window = false;

		TestGui(Window::Window* _window) :WindowGui(_window) {}
		virtual void gui()override
		{
			static float f = 0.0f;
			static int counter = 0;

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
		virtual void detach()override {}
	};
}

int main()
{
	try
	{
		printf("Fuck!\n");
		Window::Window::Data mainWindowData
		{
			"Neural Ray Tracing",
			{{1920, 1080}, /*resizable=*/true, /*fullscreen=*/false}
		};
		GUI::UserInterface ui(mainWindowData);
		OpenGL::NBody nBody1(20 * 1);
		::printf("Num particles1: %d\n", nBody1.particles.particles.length);
		ui.bindOpenGLMain(&nBody1);

		GUI::TestGui testGui(ui.mainWindow);
		ui.registerWindowGui(&testGui);

		//Window::Window::Data smallWindowData
		//{
		//	"Normal",
		//	{{400, 400}, /*resizable=*/true, /*fullscreen=*/false}
		//};
		//Window::Window& w = ui.createWindow(smallWindowData);
		//OpenGL::NBodyCUDA nBody2(10 * 1, false, String<char>("./"));
		//::printf("Num particles2: %d\n", nBody2.particles.particles.length);
		//ui.bindOpenGL(w, &nBody2);

		ui.mainLoop();
		testGui.destroy();
		return 0;
	}
	catch (const std::exception& e)
	{
		printf("%s", e.what());
		return 0;
	}
}