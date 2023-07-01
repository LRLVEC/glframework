#include <xr/_OpenXR.h>
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
		XrGraphicsBindingOpenGLWin32KHR graphicsBinding{ XR_TYPE_GRAPHICS_BINDING_OPENGL_WIN32_KHR };
		graphicsBinding.next = nullptr;
		graphicsBinding.hDC = GetDC(glfwGetWin32Window(window));
		graphicsBinding.hGLRC = glfwGetWGLContext(window);
		return graphicsBinding;
	}
#elif defined(XR_USE_PLATFORM_XLIB)
	// to do
#endif

	Math::mat4<float> get_projection(XrFovf const& fov, float nearZ, float farZ)
	{
		Math::mat4<float> result(0);
		const float tanLeft = tanf(fov.angleLeft);
		const float tanRight = tanf(fov.angleRight);
		const float tanUp = tanf(fov.angleUp);
		float const tanDown = tanf(fov.angleDown);

		const float width = tanRight - tanLeft;
		const float height = tanUp - tanDown;
		const float offsetZ = nearZ;

		result.array[0][0] = 2.0f / width;
		result.array[0][2] = (tanRight + tanLeft) / width;
		result.array[1][1] = 2.0f / height;
		result.array[1][2] = (tanUp + tanDown) / height;
		result.array[3][2] = -1.0f;

		if (farZ <= nearZ)
		{
			// place the far plane at infinity
			result.array[2][2] = -1.0f;
			result.array[2][3] = -(nearZ + offsetZ);
		}
		else
		{
			result.array[2][2] = -(farZ + offsetZ) / (farZ - nearZ);
			result.array[2][3] = -(farZ * (nearZ + offsetZ)) / (farZ - nearZ);
		}
		return result;
	}

	Math::mat4<float> get_transform(XrPosef pose, bool inv)
	{
		Math::mat4<float> result(0);
		XrQuaternionf& quat = pose.orientation;
		XrVector3f& pos = pose.position;

		if (inv)
		{
			quat.x = -quat.x;
			quat.y = -quat.y;
			quat.z = -quat.z;
		}

		const float x2 = quat.x + quat.x;
		const float y2 = quat.y + quat.y;
		const float z2 = quat.z + quat.z;

		const float xx2 = quat.x * x2;
		const float yy2 = quat.y * y2;
		const float zz2 = quat.z * z2;

		const float yz2 = quat.y * z2;
		const float wx2 = quat.w * x2;
		const float xy2 = quat.x * y2;
		const float wz2 = quat.w * z2;
		const float xz2 = quat.x * z2;
		const float wy2 = quat.w * y2;

		result.array[0][0] = 1.0f - yy2 - zz2;
		result.array[0][1] = xy2 - wz2;
		result.array[0][2] = xz2 + wy2;

		result.array[1][0] = xy2 + wz2;
		result.array[1][1] = 1.0f - xx2 - zz2;
		result.array[1][2] = yz2 - wx2;

		result.array[2][0] = xz2 - wy2;
		result.array[2][1] = yz2 + wx2;
		result.array[2][2] = 1.0f - xx2 - yy2;

		Math::vec3<float> pos_vec{ pos.x, pos.y, pos.z };
		if (inv)
			result.setCol((result, -pos_vec), 3);
		else
			result.setCol(pos_vec, 3);

		result.array[3][3] = 1.0f;
		return result;
	}

}