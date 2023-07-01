#pragma once
#include <xr/_OpenXR.h>
#include <xr/_XrVolume.h>

#include <list>
#include <stdexcept>

namespace XrOS
{
	using namespace OpenXR;
	/* XrOS is a 3D operating system that can:
	 * 1. create 3d volumes
	 * 2. run 3d apps in volumes or full space
	 * 3. handle user inputs and pass to 3d apps
	 *     trigger, pose (grip and aim pose), menu_click, others (like trackpad)
	 * 4. create color and depth render buffers for 3d apps to render
	 * 5. calculate the render viewport for 3d apps, 3d apps must render with volume bounding box
	 * 6. synthesize the color and depth render buffers by W order and submit to OpenXR
	 * 7. view VR content in the window
	 */

	 /* Abstraction:
	  * Volume Manager: create, destroy and manage volumes
	  * Action Handler: capture action from openxr and pass to volume apps
	  * World Renderer: render ui elements like bounding box and overlay volume render layers
	  * OpenXR communicator: create openxr instance, sessions and everything else
	  */


	struct VolumeManager
	{
		static VolumeManager* __volumeManager;
		std::list<Volume> volumes;
		OpenGL::Texture* colorTexture;
		OpenGL::Texture* depthTexture;
		OpenGL::TextureConfig<OpenGL::TextureStorage3D>* colorTextureConfig;
		OpenGL::TextureConfig<OpenGL::TextureStorage3D>* depthTextureConfig;


		VolumeManager()
			:
			colorTexture(nullptr),
			depthTexture(nullptr),
			colorTextureConfig(nullptr),
			depthTextureConfig(nullptr)
		{
			if (__volumeManager) throw std::runtime_error{ "Cannot construct new VolumeManager if there exists one!" };
			__volumeManager = this;
		}

		VolumeManager(VolumeCreateInfo const& _createInfo)
			:
			VolumeManager()
		{
			volumes.push_back(Volume(_createInfo));
		}

		~VolumeManager()
		{
			delete colorTexture;
			delete depthTexture;
			delete colorTextureConfig;
			delete depthTextureConfig;
		}
	};

	struct XrOS
	{
	public:
	private:
		Instance instance;
		System system;
		Session session;
		Frame frame;
		Views views;
		EventPoller poller;
		ActionSet actionSet;
		// actions


		XrOS(GLFWwindow* _xros_window)
			:
			instance("XrOS"),
			system(&instance),
			session(&system, _xros_window),
			frame(&session, XR_ENVIRONMENT_BLEND_MODE_OPAQUE),
			views(&session, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO),
			poller(&instance),
			actionSet(&instance, "xros_actions", "XrOS Actions", 0)
		{

		}

	};
}