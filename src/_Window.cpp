#include <_Window.h>

namespace Window
{
	bool Window::glewInitialized = false;
	WindowManager* WindowManager::__windowManager = nullptr;
}