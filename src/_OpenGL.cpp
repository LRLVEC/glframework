#include <_OpenGL.h>
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


namespace OpenGL
{
	bool OpenGLInit::initialized = false;
	unsigned int OpenGLInit::versionMajor = 4;
	unsigned int OpenGLInit::versionMinor = 6;
	void set_render_target_from_context(GLFWwindow* newWindow, GLFWwindow* contextWindow)
	{
#if defined(_WIN32)
		wglMakeCurrent(GetDC(glfwGetWin32Window(newWindow)), glfwGetWGLContext(contextWindow));
#elif defined(__linux__)
		glXMakeCurrent(glfwGetX11Display(), glfwGetX11Window(newWindow), glfwGetGLXContext(contextWindow));
#endif
	}
}