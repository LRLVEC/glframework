#include <_ImGui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_opengl3.h>


namespace GUI
{
	void ImGuiBase::init()
	{
		ImGui::CreateContext();
	}
}
