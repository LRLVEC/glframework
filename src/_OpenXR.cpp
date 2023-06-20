#include <_OpenXR.h>
#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#define GLFW_NATIVE_INCLUDE_NONE
#include <GLFW/glfw3native.h>
#elif defined(__linux__)
// Fuck X11
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX
#define GLFW_NATIVE_INCLUDE_NONE
#include <GLFW/glfw3native.h>
#else
static_assert("unsupported system!", true);
#endif

namespace OpenXR
{
#ifdef XR_USE_PLATFORM_WIN32
	XrGraphicsBindingOpenGLWin32KHR get_graphics_binding(GLFWwindow* window)
	{
		XrGraphicsBindingOpenGLWin32KHR graphicsBinding{XR_TYPE_GRAPHICS_BINDING_OPENGL_WIN32_KHR};
		graphicsBinding.next = nullptr;
		graphicsBinding.hDC = GetDC(glfwGetWin32Window(window));
		graphicsBinding.hGLRC = glfwGetWGLContext(window);
		return graphicsBinding;
	}
#elif defined(XR_USE_PLATFORM_XLIB)
// to do
#endif
}