#pragma once
#include <_OpenGL.h>
#include <_Texture.h>
#include <openxr/openxr.h>
// use opengl as backend
#define XR_USE_GRAPHICS_API_OPENGL
#ifdef _WIN32
#include <windows.h>
#define XR_USE_PLATFORM_WIN32 1
#endif
#if defined(OS_LINUX_XLIB)
#define XR_USE_PLATFORM_XLIB 1
#endif
#include <openxr/openxr_platform.h>
#include <openxr/openxr_reflection.h>

#include <string>
#include <vector>
#include <algorithm>


namespace OpenXR
{
#ifdef XR_USE_PLATFORM_WIN32
	XrGraphicsBindingOpenGLWin32KHR get_graphics_binding(GLFWwindow* window);
#elif defined(XR_USE_PLATFORM_XLIB)
#endif

	struct ApiLayer
	{
		std::vector<XrApiLayerProperties> layers;

		ApiLayer()
		{
			uint32_t layerCount;
			xrEnumerateApiLayerProperties(0, &layerCount, nullptr);
			layers.resize(layerCount, { XR_TYPE_API_LAYER_PROPERTIES });
			xrEnumerateApiLayerProperties((uint32_t)layers.size(), &layerCount, layers.data());
		}

		void printInfo()const
		{
			printf("Available Layers: %lld\n", layers.size());
			for (XrApiLayerProperties const& layer : layers)
			{
				printf("SpecVersion = ");
				printf("%d.%d.%d\n", XR_VERSION_MAJOR(layer.specVersion), XR_VERSION_MINOR(layer.specVersion), XR_VERSION_PATCH(layer.specVersion));
				printf("Name = %s LayerVersion = %d Description = %s\n", layer.layerName, layer.layerVersion, layer.description);
				uint32_t instanceExtensionCount;
				xrEnumerateInstanceExtensionProperties(layer.layerName, 0, &instanceExtensionCount, nullptr);
				std::vector<XrExtensionProperties> extensions(instanceExtensionCount, { XR_TYPE_EXTENSION_PROPERTIES });
				xrEnumerateInstanceExtensionProperties(layer.layerName, (uint32_t)extensions.size(), &instanceExtensionCount, extensions.data());
				char const* indentStr("    ");
				printf("%sAvailable Extensions: %d", indentStr, instanceExtensionCount);
				for (const XrExtensionProperties& extension : extensions)
					printf("%s  Name = %s SpecVersion = %d", indentStr, extension.extensionName, extension.extensionVersion);

			}
		}
	};

	struct Extension
	{
		std::vector<XrExtensionProperties> extensions;

		Extension(char const* layer_name = nullptr)
		{
			uint32_t propertyCapacity = 0;
			xrEnumerateInstanceExtensionProperties(layer_name, 0, &propertyCapacity, nullptr);
			extensions.resize(propertyCapacity, { XR_TYPE_EXTENSION_PROPERTIES });
			xrEnumerateInstanceExtensionProperties(layer_name, propertyCapacity, &propertyCapacity, extensions.data());
		}

		void printInfo()const
		{
			printf("Available Extensions: %lld\n", extensions.size());
			for (XrExtensionProperties const& extension : extensions)
			{
				printf("\t%s, ver %d%\n", extension.extensionName,
					extension.extensionVersion);
			}
		}

		bool check_extension(char const* name)const
		{
			return std::find_if(extensions.begin(), extensions.end(), [name](XrExtensionProperties const& prop)
				{
					if (std::strcmp(prop.extensionName, name) == 0)
						return true;
					return false;
				}) != extensions.end();
		}

		std::vector<bool> check_extensions(std::vector<char const*> names)const
		{
			std::vector<bool> valid(names.size());
			std::transform(names.begin(), names.end(), valid.begin(),
				[&](char const* name) {return check_extension(name); });
			return valid;
		}
	};

	struct Instance
	{
		XrInstance instance;
		Extension extension;
		std::vector<char const*> validExtensions;
		bool enabledDepth;

		Instance(char const* app_name = nullptr, std::vector<char const*> extraExtensions = {}, bool _enableDepth = false)
			:
			instance(XR_NULL_HANDLE),
			validExtensions({ XR_KHR_OPENGL_ENABLE_EXTENSION_NAME }),
			enabledDepth(false)
		{
			if (!extension.check_extension(XR_KHR_OPENGL_ENABLE_EXTENSION_NAME))
			{
				printf("Error! Platform doesn't support OpenGL extension!\n");
				return;
			}

			// not enable XR_KHR_composition_layer_depth by default
			// since there may be performance issue when enabled
			if (_enableDepth)
				extraExtensions.push_back(XR_KHR_COMPOSITION_LAYER_DEPTH_EXTENSION_NAME);
			if (extraExtensions.size())
			{
				std::vector<bool> valid = extension.check_extensions(extraExtensions);
				for (uint32_t c0(0); c0 < extraExtensions.size(); ++c0)
				{
					if (valid[c0])
						validExtensions.push_back(extraExtensions[c0]);
				}
			}
			if (std::find_if(validExtensions.begin(), validExtensions.end(), [](char const* ext) {
				if (std::strcmp(ext, XR_KHR_COMPOSITION_LAYER_DEPTH_EXTENSION_NAME) == 0)
					return true;
				return false; }) != validExtensions.end())
			{
				enabledDepth = true;
			}

			XrInstanceCreateInfo instance_create_info{ XR_TYPE_INSTANCE_CREATE_INFO };
			instance_create_info.next = nullptr;
			instance_create_info.enabledExtensionCount = validExtensions.size();
			instance_create_info.enabledExtensionNames = validExtensions.data();
			instance_create_info.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
			if (app_name)
				strcpy(instance_create_info.applicationInfo.applicationName, app_name);
			xrCreateInstance(&instance_create_info, &instance);
			if (instance == XR_NULL_HANDLE)
			{
				printf("failed to create xr instance %s\n", app_name);
			}
		}
		Instance(Instance const&) = delete;
		Instance(Instance&& _i) noexcept
			:
			instance(_i.instance)
		{
			_i.instance = XR_NULL_HANDLE;
		}

		~Instance()
		{
			if (instance != XR_NULL_HANDLE)
			{
				xrDestroyInstance(instance);
				instance = XR_NULL_HANDLE;
			}
		}

		operator bool()const
		{
			return instance != XR_NULL_HANDLE;
		}

		operator XrInstance()const
		{
			return instance;
		}

		void printInfo()const
		{
			XrInstanceProperties instance_properties{ XR_TYPE_INSTANCE_PROPERTIES };
			xrGetInstanceProperties(instance, &instance_properties);
			XrVersion ver = instance_properties.runtimeVersion;
			printf("Instance Runtime Name = %s\n", instance_properties.runtimeName);
			printf("Instance Runtime Version = %d.%d.%d\n", XR_VERSION_MAJOR(ver), XR_VERSION_MINOR(ver), XR_VERSION_PATCH(ver));
			printf("Enabled Extensions: %d\n", validExtensions.size());
			for (auto name : validExtensions)
			{
				printf("\t%s\n", name);
			}
		}

	};

	struct EventPoller
	{
		XrEventDataBuffer eventDataBuffer;
		Instance* instance;

		EventPoller(Instance* _instance)
			:
			eventDataBuffer{ XR_TYPE_EVENT_DATA_BUFFER },
			instance(_instance)
		{
		}
		EventPoller(EventPoller const&) = delete;
		EventPoller(EventPoller&& _e) noexcept
			:
			eventDataBuffer(_e.eventDataBuffer),
			instance(_e.instance)
		{
			_e.eventDataBuffer = { XR_TYPE_EVENT_DATA_BUFFER };
			_e.instance = nullptr;
		}

		XrEventDataBaseHeader const* poll()
		{
			XrEventDataBaseHeader* baseHeader = reinterpret_cast<XrEventDataBaseHeader*>(&eventDataBuffer);
			*baseHeader = { XR_TYPE_EVENT_DATA_BUFFER };
			XrResult const result = xrPollEvent(*instance, &eventDataBuffer);
			if (result == XR_SUCCESS)
			{
				if (baseHeader->type == XR_TYPE_EVENT_DATA_EVENTS_LOST)
				{
					XrEventDataEventsLost const* eventsLost = reinterpret_cast<XrEventDataEventsLost const*>(baseHeader);
					printf("%u events lost", eventsLost->lostEventCount);
				}
				return baseHeader;
			}
			if (result == XR_EVENT_UNAVAILABLE)
				return nullptr;
			printf("xrPollEvent error: %d\n", result);
			return nullptr;
		}
	};

	struct System
	{
		XrSystemId systemId;
		Instance* instance;
		PFN_xrGetOpenGLGraphicsRequirementsKHR pfnGetOpenGLGraphicsRequirementsKHR;

		System(Instance* _instance)
			:
			systemId(XR_NULL_SYSTEM_ID),
			instance(_instance),
			pfnGetOpenGLGraphicsRequirementsKHR(nullptr)
		{
			XrSystemGetInfo system_info{ XR_TYPE_SYSTEM_GET_INFO };
			system_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY; // for hmd
			xrGetSystem(*_instance, &system_info, &systemId);
			if (systemId == XR_NULL_SYSTEM_ID)
			{
				printf("failed to get xr system!\n");
				return;
			}
			xrGetInstanceProcAddr(*instance, "xrGetOpenGLGraphicsRequirementsKHR",
				reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetOpenGLGraphicsRequirementsKHR));
			if (!pfnGetOpenGLGraphicsRequirementsKHR)
			{
				printf("failed to get xrGetOpenGLGraphicsRequirementsKHR!\n");
				return;
			}
			XrGraphicsRequirementsOpenGLKHR graphics_requirements{ XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_KHR };
			pfnGetOpenGLGraphicsRequirementsKHR(*instance, systemId, &graphics_requirements);
			XrVersion ver = graphics_requirements.minApiVersionSupported;
			printf("Min Api Version Supported:  %d.%d.%d\n", XR_VERSION_MAJOR(ver), XR_VERSION_MINOR(ver), XR_VERSION_PATCH(ver));
		}
		System(System&& _s) noexcept
			:
			systemId(_s.systemId),
			instance(_s.instance),
			pfnGetOpenGLGraphicsRequirementsKHR(_s.pfnGetOpenGLGraphicsRequirementsKHR)
		{
			_s.systemId = XR_NULL_SYSTEM_ID;
			_s.instance = nullptr;
			_s.pfnGetOpenGLGraphicsRequirementsKHR = nullptr;
		}

		~System()
		{
			systemId = XR_NULL_SYSTEM_ID;
			instance = nullptr;
			pfnGetOpenGLGraphicsRequirementsKHR = nullptr;
		}

		operator bool()const
		{
			return systemId != XR_NULL_SYSTEM_ID;
		}

		operator XrSystemId()const
		{
			return systemId;
		}

		operator XrInstance()const
		{
			return instance->instance;
		}
	};

	struct Session
	{
		XrSession session;
		XrSessionState state;
		System* system;
		GLFWwindow* window;

		Session(System* _system, GLFWwindow* _window)
			:
			session(XR_NULL_HANDLE),
			state(XR_SESSION_STATE_UNKNOWN),
			system(_system),
			window(_window)
		{
			XrGraphicsBindingOpenGLWin32KHR graphics_binding = get_graphics_binding(_window);
			XrSessionCreateInfo session_create_info{ XR_TYPE_SESSION_CREATE_INFO };
			session_create_info.next = reinterpret_cast<const XrBaseInStructure*>(&graphics_binding);
			session_create_info.systemId = *system;
			xrCreateSession(*system, &session_create_info, &session);
			if (session == XR_NULL_HANDLE)
			{
				printf("failed to create xr session!\n");
			}
		}
		Session(Session const&) = delete;
		Session(Session&& _s) noexcept
			:
			session(_s.session),
			state(_s.state),
			system(_s.system),
			window(_s.window)
		{
			_s.session = XR_NULL_HANDLE;
			_s.state = XR_SESSION_STATE_UNKNOWN;
			_s.system = nullptr;
			_s.window = nullptr;
		}

		~Session()
		{
			if (session != XR_NULL_HANDLE)
			{
				xrDestroySession(session);
				session = XR_NULL_HANDLE;
				state = XR_SESSION_STATE_UNKNOWN;
				system = nullptr;
				window = nullptr;
			}
		}

		operator bool()const
		{
			return session != XR_NULL_HANDLE;
		}

		operator XrSession()const
		{
			return session;
		}

		operator XrSystemId()const
		{
			return system->systemId;
		}

		operator XrInstance()const
		{
			return system->instance->instance;
		}

		void printSystemInfo()const
		{
			XrSystemProperties systemProperties{ XR_TYPE_SYSTEM_PROPERTIES };
			xrGetSystemProperties(*system, *system, &systemProperties);
			printf("System Properties: Name=%s VendorId=%d\n", systemProperties.systemName, systemProperties.vendorId);
			printf("System Graphics Properties: MaxWidth=%d MaxHeight=%d MaxLayers=%d\n",
				systemProperties.graphicsProperties.maxSwapchainImageWidth,
				systemProperties.graphicsProperties.maxSwapchainImageHeight,
				systemProperties.graphicsProperties.maxLayerCount);
			printf("System Tracking Properties: OrientationTracking=%s PositionTracking=%s\n",
				systemProperties.trackingProperties.orientationTracking == XR_TRUE ? "True" : "False",
				systemProperties.trackingProperties.positionTracking == XR_TRUE ? "True" : "False");
		}

		void printReferenceSpace()const
		{
			uint32_t spaceCount;
			xrEnumerateReferenceSpaces(session, 0, &spaceCount, nullptr);
			std::vector<XrReferenceSpaceType> spaces(spaceCount);
			xrEnumerateReferenceSpaces(session, spaceCount, &spaceCount, spaces.data());
			printf("Available reference spaces: %d\n", spaceCount);
			for (XrReferenceSpaceType space : spaces)
			{
				printf("Name: ");
				switch (space)
				{
				case XR_REFERENCE_SPACE_TYPE_VIEW: printf("View\n"); break;
				case XR_REFERENCE_SPACE_TYPE_LOCAL: printf("Local\n"); break;
				case XR_REFERENCE_SPACE_TYPE_STAGE: printf("Stage\n"); break;
				case XR_REFERENCE_SPACE_TYPE_UNBOUNDED_MSFT: printf("Unbounded MSFT\n"); break;
				case XR_REFERENCE_SPACE_TYPE_COMBINED_EYE_VARJO: printf("Combined Eye Varjo\n"); break;
				}
			}
		}

	};

	struct Frame
	{
		XrFrameState frameState;
		XrEnvironmentBlendMode envBlendMode;
		Session* session;

		Frame(Session* _session, XrEnvironmentBlendMode _envBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE)
			:
			frameState{ XR_TYPE_FRAME_STATE },
			envBlendMode(_envBlendMode),
			session(_session)
		{

		}
		Frame(Frame const&) = delete;
		Frame(Frame&& _f) noexcept
			:
			frameState(_f.frameState),
			envBlendMode(_f.envBlendMode),
			session(_f.session)
		{
			_f.frameState = { XR_TYPE_FRAME_STATE };
			_f.envBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
			_f.session = nullptr;
		}

		~Frame()
		{
			frameState = { XR_TYPE_FRAME_STATE };
			session = nullptr;
		}

		bool should_render()const
		{
			return frameState.shouldRender;
		}

		XrTime predicted_display_time()const
		{
			return frameState.predictedDisplayTime;
		}

		XrDuration predicted_display_period()const
		{
			return frameState.predictedDisplayPeriod;
		}

		void wait()
		{
			XrFrameWaitInfo frameWaitInfo{ XR_TYPE_FRAME_WAIT_INFO };
			xrWaitFrame(*session, &frameWaitInfo, &frameState);
		}

		void begin()
		{
			XrFrameBeginInfo frameBeginInfo{ XR_TYPE_FRAME_BEGIN_INFO };
			xrBeginFrame(*session, &frameBeginInfo);
		}

		void end(std::vector<XrCompositionLayerBaseHeader*>const& layers)
		{
			XrFrameEndInfo frameEndInfo{ XR_TYPE_FRAME_END_INFO };
			frameEndInfo.displayTime = frameState.predictedDisplayTime;
			frameEndInfo.environmentBlendMode = envBlendMode;
			frameEndInfo.layerCount = layers.size();
			frameEndInfo.layers = layers.data();
			xrEndFrame(*session, &frameEndInfo);
		}
	};

	struct Swapchain
	{
		XrSwapchain swapchain;
		int32_t width;
		int32_t height;
		Session* session;

		Swapchain(Session* _session, int64_t _colorSwapchainFormat, XrViewConfigurationView const& _view_config, uint64_t _usage_flags)
			:
			swapchain(XR_NULL_HANDLE),
			width(0),
			height(0),
			session(_session)
		{
			XrSwapchainCreateInfo swapchainCreateInfo{ XR_TYPE_SWAPCHAIN_CREATE_INFO };
			swapchainCreateInfo.usageFlags = _usage_flags;
			swapchainCreateInfo.format = _colorSwapchainFormat;
			swapchainCreateInfo.sampleCount = _view_config.recommendedSwapchainSampleCount;
			swapchainCreateInfo.width = _view_config.recommendedImageRectWidth;
			swapchainCreateInfo.height = _view_config.recommendedImageRectHeight;
			swapchainCreateInfo.faceCount = 1;
			swapchainCreateInfo.arraySize = 1;
			swapchainCreateInfo.mipCount = 1;
			width = swapchainCreateInfo.width;
			height = swapchainCreateInfo.height;
			xrCreateSwapchain(*session, &swapchainCreateInfo, &swapchain);
		}
		Swapchain(Swapchain const&) = delete;
		Swapchain(Swapchain&& _s)noexcept
			:
			swapchain(_s.swapchain),
			width(_s.width),
			height(_s.height),
			session(_s.session)
		{
			_s.swapchain = XR_NULL_HANDLE;
			_s.width = 0;
			_s.height = 0;
			_s.session = nullptr;
		}

		~Swapchain()
		{
			if (swapchain != XR_NULL_HANDLE)
			{
				xrDestroySwapchain(swapchain);
				width = height = 0;
				session = nullptr;
			}
		}

		operator bool()const
		{
			return swapchain != XR_NULL_HANDLE;
		}

		operator XrSwapchain()const
		{
			return swapchain;
		}

		operator XrSession()const
		{
			return session->session;
		}

		void acquire(uint32_t* swapchainImageIndex)
		{
			XrSwapchainImageAcquireInfo acquireInfo{ XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
			xrAcquireSwapchainImage(swapchain, &acquireInfo, swapchainImageIndex);
		}

		void wait(XrDuration timeout = XR_INFINITE_DURATION)
		{
			XrSwapchainImageWaitInfo waitInfo{ XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
			waitInfo.timeout = timeout;
			xrWaitSwapchainImage(swapchain, &waitInfo);
		}

		void release()
		{
			XrSwapchainImageReleaseInfo releaseInfo{ XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
			xrReleaseSwapchainImage(swapchain, &releaseInfo);
		}
	};

	struct SwapchainImages : Swapchain
	{
		std::vector<XrSwapchainImageOpenGLKHR> swapchainImages;

		SwapchainImages(Session* _session, int64_t _colorSwapchainFormat, XrViewConfigurationView const& _view_config, uint64_t _usage_flags)
			:
			Swapchain(_session, _colorSwapchainFormat, _view_config, _usage_flags)
		{
			uint32_t imageCount;
			xrEnumerateSwapchainImages(swapchain, 0, &imageCount, nullptr);
			swapchainImages.resize(imageCount, { XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR });
			xrEnumerateSwapchainImages(swapchain, imageCount, &imageCount, reinterpret_cast<XrSwapchainImageBaseHeader*>(swapchainImages.data()));
		}

		void printInfo()const
		{
			printf("SwapchainImages has %lld images.\n", swapchainImages.size());
		}
	};

	struct Views
	{
		int64_t colorSwapchainFormat;
		int64_t depthSwapchainFormat;
		XrViewConfigurationType viewType;
		uint32_t validViewCount;
		std::vector<XrView> views;
		std::vector<XrViewConfigurationType> validViewType;
		std::vector<XrViewConfigurationView> viewConfigs;
		std::vector<SwapchainImages> colorSwapchainImages;
		std::vector<SwapchainImages> depthSwapchainImages;
		Session* session;

		Views(Session* _session, XrViewConfigurationType _view_config_type = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO)
			:
			colorSwapchainFormat(-1),
			depthSwapchainFormat(-1),
			viewType(_view_config_type),
			validViewCount(0),
			session(_session)
		{
			uint32_t view_config_type_count;
			xrEnumerateViewConfigurations(*session, *session, 0, &view_config_type_count, nullptr);
			validViewType.resize(view_config_type_count);
			xrEnumerateViewConfigurations(*session, *session, view_config_type_count, &view_config_type_count, validViewType.data());

			// fallback to stereo or mono
			auto begin = validViewType.begin();
			auto end = validViewType.end();
			if (std::find(begin, end, viewType) == end)
			{
				printf("Warning! Cannot find view config type [%s]!\n", get_str(viewType));
				if (std::find(begin, end, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO) != end)
					viewType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
				else if (std::find(begin, end, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_MONO) != end)
					viewType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_MONO;
				else
				{
					printf("Cannot find suitable config type!\n");
					return;
				}
				printf("Use [%s] instead.\n", get_str(viewType));
			}

			uint32_t view_count(0);
			xrEnumerateViewConfigurationViews(*session, *session, viewType, 0, &view_count, nullptr);
			viewConfigs.resize(view_count, { XR_TYPE_VIEW_CONFIGURATION_VIEW });
			views.resize(view_count, { XR_TYPE_VIEW });
			xrEnumerateViewConfigurationViews(*session, *session, viewType, view_count, &view_count, viewConfigs.data());

			select_swapchain_formats();
			for (uint32_t i(0); i < view_count; ++i)
				colorSwapchainImages.push_back(SwapchainImages(session, colorSwapchainFormat, viewConfigs[i],
					XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT));
			if (enabledDepth())
			{
				for (uint32_t i(0); i < view_count; ++i)
					depthSwapchainImages.push_back(SwapchainImages(session, depthSwapchainFormat, viewConfigs[i],
						XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT));
			}
		}

		bool enabledDepth()const
		{
			return session->system->instance->enabledDepth;
		}

		void select_swapchain_formats()
		{
			uint32_t swapchainFormatCount;
			xrEnumerateSwapchainFormats(*session, 0, &swapchainFormatCount, nullptr);
			std::vector<int64_t> swapchainFormats(swapchainFormatCount);
			xrEnumerateSwapchainFormats(*session, (uint32_t)swapchainFormats.size(), &swapchainFormatCount, swapchainFormats.data());

			constexpr OpenGL::TextureFormat SupportedColorSwapchainFormats[] =
			{
				OpenGL::TextureFormat::RGB10A2,
				OpenGL::TextureFormat::RGBA16f,
				OpenGL::TextureFormat::RGBA8,
				OpenGL::TextureFormat::RGBA8Snorm,
			};
			for (auto fmt : SupportedColorSwapchainFormats)
			{
				bool found(std::find(swapchainFormats.begin(), swapchainFormats.end(), fmt) != swapchainFormats.end());
				if (found)
				{
					colorSwapchainFormat = (int64_t)fmt;
					break;
				}
			}

			if (enabledDepth())
			{
				constexpr int64_t SupportedDepthSwapchainFormats[] =
				{
					GL_DEPTH_COMPONENT32F,
					GL_DEPTH_COMPONENT24,
					GL_DEPTH_COMPONENT16,
				};
				for (auto fmt : SupportedDepthSwapchainFormats)
				{
					bool found(std::find(swapchainFormats.begin(), swapchainFormats.end(), fmt) != swapchainFormats.end());
					if (found)
					{
						depthSwapchainFormat = fmt;
						break;
					}
				}
			}
		}

		bool locate(XrTime predictedDisplayTime, XrSpace space)
		{
			XrViewState viewState{ XR_TYPE_VIEW_STATE };
			uint32_t viewCapacityInput = views.size();

			XrViewLocateInfo viewLocateInfo{ XR_TYPE_VIEW_LOCATE_INFO };
			viewLocateInfo.viewConfigurationType = viewType;
			viewLocateInfo.displayTime = predictedDisplayTime;
			viewLocateInfo.space = space;

			xrLocateViews(*session, &viewLocateInfo, &viewState, viewCapacityInput, &validViewCount, views.data());
			if (!(viewState.viewStateFlags & (XR_VIEW_STATE_POSITION_VALID_BIT | XR_VIEW_STATE_ORIENTATION_VALID_BIT)))
				return false;  // no valid tracking poses for the views
			if (validViewCount != views.size())
				return false;
			return true;
		}

		char const* get_str(XrViewConfigurationType _view_config_type)const
		{
			switch (_view_config_type)
			{
			case XR_VIEW_CONFIGURATION_TYPE_PRIMARY_MONO: return "Mono";
			case XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO: return "Stereo";
			case XR_VIEW_CONFIGURATION_TYPE_PRIMARY_QUAD_VARJO: return "Quad Varjo";
			case XR_VIEW_CONFIGURATION_TYPE_SECONDARY_MONO_FIRST_PERSON_OBSERVER_MSFT: return "MONO_FIRST_PERSON_OBSERVER_MSFT";
			}
		}

		void printViewInfo()const
		{
			printf("Available View Config Types: %lld\n", validViewType.size());

			for (XrViewConfigurationType view_config_type : validViewType)
			{
				printf("View Config Type: [%s]%s\n", get_str(view_config_type), view_config_type == viewType ? " (selected) " : "");

				XrViewConfigurationProperties viewConfigProperties{ XR_TYPE_VIEW_CONFIGURATION_PROPERTIES };
				xrGetViewConfigurationProperties(*session, *session, view_config_type, &viewConfigProperties);
				printf("View Config FovMutable: %s\n", viewConfigProperties.fovMutable == XR_TRUE ? "True" : "False");

				uint32_t view_count(0);
				xrEnumerateViewConfigurationViews(*session, *session, view_config_type, 0, &view_count, nullptr);
				if (view_count > 0)
				{
					std::vector<XrViewConfigurationView> views(view_count, { XR_TYPE_VIEW_CONFIGURATION_VIEW });
					xrEnumerateViewConfigurationViews(*session, *session, view_config_type, view_count, &view_count, views.data());

					for (uint32_t i = 0; i < views.size(); i++)
					{
						const XrViewConfigurationView& view = views[i];
						printf("View %d: Recommended Width=%d Height=%d SampleCount=%d\n", i,
							view.recommendedImageRectWidth,
							view.recommendedImageRectHeight,
							view.recommendedSwapchainSampleCount);
						printf("View %d:     Maximum Width=%d Height=%d SampleCount=%d\n", i, view.maxImageRectWidth,
							view.maxImageRectHeight, view.maxSwapchainSampleCount);
					}
				}
				else
				{
					printf("Empty view configuration type\n");
				}

				uint32_t count(0);
				xrEnumerateEnvironmentBlendModes(*session, *session, view_config_type, 0, &count, nullptr);
				printf("Available Environment Blend Mode count: %d\n", count);
				std::vector<XrEnvironmentBlendMode> blend_modes(count);
				xrEnumerateEnvironmentBlendModes(*session, *session, view_config_type, count, &count, blend_modes.data());
				for (XrEnvironmentBlendMode mode : blend_modes)
				{
					printf("Environment Blend Mode: ");
					switch (mode)
					{
					case XR_ENVIRONMENT_BLEND_MODE_OPAQUE: printf("Opaque\n"); break;
					case XR_ENVIRONMENT_BLEND_MODE_ADDITIVE: printf("Additive\n"); break;
					case XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND: printf("Alpha Blend\n"); break;
					}
				}
			}
		}

		void printViewConfig()const
		{
			for (uint32_t i(0); i < viewConfigs.size(); ++i)
			{
				auto const& view_config = viewConfigs[i];
				printf("Swapchain for view %d: Width = %d Height = %d SampleCount = %d\n", i,
					view_config.recommendedImageRectWidth,
					view_config.recommendedImageRectHeight,
					view_config.recommendedSwapchainSampleCount);
			}
		}

		void printSwapchainInfo()const
		{
			printf("selected color swapchain format: ");
			switch (colorSwapchainFormat)
			{
			case -1: printf("No valid format!\n"); break;
			case OpenGL::TextureFormat::RGB10A2: printf("RGB10A2\n"); break;
			case OpenGL::TextureFormat::RGBA16f: printf("RGBA16f\n"); break;
			case OpenGL::TextureFormat::RGBA8: printf("RGBA8\n"); break;
			case OpenGL::TextureFormat::RGBA8Snorm: printf("RGBA8Snorm\n"); break;
			default: printf("Unknown!\n"); break;
			}
			for (uint32_t i(0); i < colorSwapchainImages.size(); ++i)
			{
				SwapchainImages const& swapchainImage = colorSwapchainImages[i];
				swapchainImage.printInfo();
			}
			if (enabledDepth())
			{
				printf("selected depth swapchain format: ");
				switch (depthSwapchainFormat)
				{
				case -1: printf("No valid format!\n"); break;
				case GL_DEPTH_COMPONENT32F: printf("Component32F\n"); break;
				case GL_DEPTH_COMPONENT24: printf("Component24\n"); break;
				case GL_DEPTH_COMPONENT16: printf("Component16\n"); break;
				default: printf("Unknown!\n"); break;
				}
				for (uint32_t i(0); i < depthSwapchainImages.size(); ++i)
				{
					SwapchainImages const& swapchainImage = depthSwapchainImages[i];
					swapchainImage.printInfo();
				}
			}
		}
	};

	struct Path
	{
		XrPath path;
		Instance* instance;

		Path() :path(XR_NULL_PATH), instance(nullptr) {}
		Path(Instance* _instance, char const* str = nullptr)
			:
			path(XR_NULL_PATH),
			instance(_instance)
		{
			if (str)
				xrStringToPath(*instance, str, &path);
		}

		~Path()
		{
			if (path)
			{
				path = XR_NULL_PATH;
				instance = nullptr;
			}
		}

		operator XrPath()const
		{
			return path;
		}

		bool operator==(Path const& _p)const
		{
			return (path == _p.path) && (instance == _p.instance);
		}

		bool operator==(XrPath const& _p)const
		{
			return path == _p;
		}

		void create(char const* str)
		{
			if (str && instance)
				xrStringToPath(*instance, str, &path);
		}
	};

	struct Action;

	struct ActionSet
	{
		XrActionSet actionSet;
		Instance* instance;
		Session* session;
		uint32_t priority;
		std::vector<Action*> actions;

		ActionSet(Instance* _instance, char const* name, char const* localized_name, uint32_t _priority = 0)
			:
			actionSet(XR_NULL_HANDLE),
			instance(_instance),
			session(nullptr),
			priority(_priority)
		{
			XrActionSetCreateInfo action_set_info{ XR_TYPE_ACTION_SET_CREATE_INFO };
			strcpy_s(action_set_info.actionSetName, name);
			strcpy_s(action_set_info.localizedActionSetName, localized_name);
			action_set_info.priority = 0;
			xrCreateActionSet(*_instance, &action_set_info, &actionSet);
			if (actionSet == XR_NULL_HANDLE)
			{
				printf("failed to create xr actionSet %s!\n", name);
			}
		}

		~ActionSet()
		{
			if (actionSet != XR_NULL_HANDLE)
			{
				xrDestroyActionSet(actionSet);
				actionSet = XR_NULL_HANDLE;
				instance = nullptr;
				priority = 0;
			}
		}

		operator bool()const
		{
			return actionSet != XR_NULL_HANDLE;
		}

		operator XrActionSet()const
		{
			return actionSet;
		}

		operator XrInstance()const
		{
			return instance->instance;
		}

		void attach_session(Session* _session)
		{
			session = _session;
			XrSessionActionSetsAttachInfo attachInfo{ XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO };
			attachInfo.countActionSets = 1;
			attachInfo.actionSets = &actionSet;
			xrAttachSessionActionSets(*session, &attachInfo);
		}

		void sync()
		{
			const XrActiveActionSet activeActionSet{ actionSet, XR_NULL_PATH };
			XrActionsSyncInfo syncInfo{ XR_TYPE_ACTIONS_SYNC_INFO };
			syncInfo.countActiveActionSets = 1;
			syncInfo.activeActionSets = &activeActionSet;
			xrSyncActions(*session, &syncInfo);
		}
	};

	struct Action
	{
		XrAction action;
		XrActionType actionType;
		ActionSet* actionSet;
		std::vector<XrPath> subactionPaths;
		std::string name;

		Action()
			:
			action(XR_NULL_HANDLE),
			actionType(XR_ACTION_TYPE_MAX_ENUM),
			actionSet(nullptr),
			name()
		{
		}
		Action(ActionSet* _actionSet, char const* _name, char const* localized_name,
			XrActionType _action_type, std::vector<Path> const& subaction_paths = {})
			:
			action(XR_NULL_HANDLE),
			actionType(_action_type),
			actionSet(_actionSet),
			name(_name)
		{
			XrActionCreateInfo actionInfo{ XR_TYPE_ACTION_CREATE_INFO };
			subactionPaths.resize(subaction_paths.size());
			std::transform(subaction_paths.begin(), subaction_paths.end(),
				subactionPaths.begin(), [](Path const& _path) {return _path.path; });

			strcpy_s(actionInfo.actionName, _name);
			strcpy_s(actionInfo.localizedActionName, localized_name);

			actionInfo.actionType = _action_type;
			actionInfo.countSubactionPaths = subaction_paths.size();
			actionInfo.subactionPaths = subactionPaths.data();

			xrCreateAction(_actionSet->actionSet, &actionInfo, &action);
			if (action == XR_NULL_HANDLE)
			{
				printf("failed to create xr action %s!\n", _name);
				return;
			}
			actionSet->actions.push_back(this);
		}
		Action(Action const&) = delete;
		Action(Action&& _a) noexcept
			:
			action(_a.action),
			actionType(_a.actionType),
			actionSet(_a.actionSet),
			name(_a.name)
		{
			_a.action = XR_NULL_HANDLE;
			_a.actionType = XR_ACTION_TYPE_MAX_ENUM;
			_a.actionSet = nullptr;
			_a.name.clear();
		}

		~Action()
		{
			if (action != XR_NULL_HANDLE)
			{
				xrDestroyAction(action);
				action = XR_NULL_HANDLE;
				actionType = XR_ACTION_TYPE_MAX_ENUM;
				actionSet = nullptr;
				name.clear();
			}
		}

		operator bool()const
		{
			return action != XR_NULL_HANDLE;
		}

		operator XrAction()const
		{
			return action;
		}

		operator XrActionSet()const
		{
			return actionSet->actionSet;
		}

		operator XrSession()const
		{
			return actionSet->session->session;
		}
	};

	enum class ActionType
	{
		BoolInput = XR_ACTION_TYPE_BOOLEAN_INPUT,
		FloatInput = XR_ACTION_TYPE_FLOAT_INPUT,
		Vec2fInput = XR_ACTION_TYPE_VECTOR2F_INPUT,
		PoseInput = XR_ACTION_TYPE_POSE_INPUT,
		VibrationOutput = XR_ACTION_TYPE_VIBRATION_OUTPUT,
	};

	struct ActionStateBase
	{
		Action* action;
		Path* subactionPath;

		ActionStateBase(Action* _action, Path* _subactionPath = nullptr)
			:
			action(_action),
			subactionPath(_subactionPath)
		{
			auto const& paths = action->subactionPaths;
			if (paths.size())
			{
				if (!subactionPath)
				{
					printf("Cannot create ActionState without subactionPath for action with subactionPaths!\n");
				}
				else if (std::find_if(paths.begin(), paths.end(), [&](XrPath const& _p) { return subactionPath->path == _p; }) == paths.end())
				{
					printf("Cannot create ActionState with subactionPath not in action subactionPaths list!\n");
				}
			}
		}
		virtual void update_state() = 0;

		virtual bool is_active()const = 0;

		virtual bool changed_since_last_sync()const = 0;

		virtual XrTime last_change_time()const = 0;

		virtual void printInfo()const {}
	};

	template<ActionType _actionType>struct ActionState : ActionStateBase {};

	template<>struct ActionState<ActionType::BoolInput> : ActionStateBase
	{
		using ret_type = XrBool32;
		static constexpr XrActionType _ActionType = XR_ACTION_TYPE_BOOLEAN_INPUT;
		static constexpr XrStructureType _ActionStateType = XR_TYPE_ACTION_STATE_BOOLEAN;

		XrActionStateBoolean state;

		ActionState(Action* _action, Path* _subactionPath = nullptr)
			:
			ActionStateBase(_action, _subactionPath),
			state{ _ActionStateType }
		{
		}

		void update_state()override
		{
			XrActionStateGetInfo getInfo{ XR_TYPE_ACTION_STATE_GET_INFO, nullptr, *action, XR_NULL_PATH };
			if (subactionPath)
				getInfo.subactionPath = *subactionPath;
			state = { _ActionStateType };
			xrGetActionStateBoolean(*action, &getInfo, &state);
		}

		ret_type current_state()const
		{
			return state.currentState;
		}

		bool is_active()const override
		{
			return state.isActive;
		}

		bool changed_since_last_sync()const override
		{
			return state.changedSinceLastSync;
		}

		XrTime last_change_time()const override
		{
			return state.lastChangeTime;
		}

		void printInfo()const override
		{
			printf("ActionState %s:\n", action->name.c_str());
			printf("\tstate: %s\n", state.currentState ? "True" : "False");
			printf("\tis active: %s\n", state.isActive ? "True" : "False");
			printf("\tchanged since last sync: %s\n", state.changedSinceLastSync ? "True" : "False");
			printf("\tlast change time: %lld\n", state.lastChangeTime);
		}
	};
	template<>struct ActionState<ActionType::FloatInput> : ActionStateBase
	{
		using ret_type = float;
		static constexpr XrActionType _ActionType = XR_ACTION_TYPE_FLOAT_INPUT;
		static constexpr XrStructureType _ActionStateType = XR_TYPE_ACTION_STATE_FLOAT;

		XrActionStateFloat state;

		ActionState(Action* _action, Path* _subactionPath = nullptr)
			:
			ActionStateBase(_action, _subactionPath),
			state{ _ActionStateType }
		{
		}

		void update_state() override
		{
			XrActionStateGetInfo getInfo{ XR_TYPE_ACTION_STATE_GET_INFO, nullptr, *action, XR_NULL_PATH };
			if (subactionPath)
				getInfo.subactionPath = *subactionPath;
			state = { _ActionStateType };
			xrGetActionStateFloat(*action, &getInfo, &state);
		}

		ret_type current_state()const
		{
			return state.currentState;
		}

		bool is_active()const override
		{
			return state.isActive;
		}

		bool changed_since_last_sync()const override
		{
			return state.changedSinceLastSync;
		}

		XrTime last_change_time()const override
		{
			return state.lastChangeTime;
		}

		void printInfo()const override
		{
			printf("ActionState %s:\n", action->name.c_str());
			printf("\tstate: %f\n", state.currentState);
			printf("\tis active: %s\n", state.isActive ? "True" : "False");
			printf("\tchanged since last sync: %s\n", state.changedSinceLastSync ? "True" : "False");
			printf("\tlast change time: %lld\n", state.lastChangeTime);
		}
	};
	template<>struct ActionState<ActionType::Vec2fInput> : ActionStateBase
	{
		using ret_type = XrVector2f;
		static constexpr XrActionType _ActionType = XR_ACTION_TYPE_VECTOR2F_INPUT;
		static constexpr XrStructureType _ActionStateType = XR_TYPE_ACTION_STATE_VECTOR2F;

		XrActionStateVector2f state;

		ActionState(Action* _action, Path* _subactionPath = nullptr)
			:
			ActionStateBase(_action, _subactionPath),
			state{ _ActionStateType }
		{
		}

		void update_state() override
		{
			XrActionStateGetInfo getInfo{ XR_TYPE_ACTION_STATE_GET_INFO, nullptr, *action, XR_NULL_PATH };
			if (subactionPath)
				getInfo.subactionPath = *subactionPath;
			state = { _ActionStateType };
			xrGetActionStateVector2f(*action, &getInfo, &state);
		}

		ret_type current_state()const
		{
			return state.currentState;
		}

		bool is_active()const override
		{
			return state.isActive;
		}

		bool changed_since_last_sync()const override
		{
			return state.changedSinceLastSync;
		}

		XrTime last_change_time()const override
		{
			return state.lastChangeTime;
		}

		void printInfo()const override
		{
			printf("ActionState %s:\n", action->name.c_str());
			printf("\tstate: [%f, %f]\n", state.currentState.x, state.currentState.y);
			printf("\tis active: %s\n", state.isActive ? "True" : "False");
			printf("\tchanged since last sync: %s\n", state.changedSinceLastSync ? "True" : "False");
			printf("\tlast change time: %lld\n", state.lastChangeTime);
		}
	};
	template<>struct ActionState<ActionType::PoseInput> : ActionStateBase
	{
		using ret_type = XrPosef;
		static constexpr XrActionType _ActionType = XR_ACTION_TYPE_POSE_INPUT;
		static constexpr XrStructureType _ActionStateType = XR_TYPE_ACTION_STATE_POSE;

		XrActionStatePose state;

		ActionState(Action* _action, Path* _subactionPath = nullptr)
			:
			ActionStateBase(_action, _subactionPath),
			state{ _ActionStateType }
		{
		}

		void update_state() override
		{
			XrActionStateGetInfo getInfo{ XR_TYPE_ACTION_STATE_GET_INFO, nullptr, *action, XR_NULL_PATH };
			if (subactionPath)
				getInfo.subactionPath = *subactionPath;
			state = { _ActionStateType };
			xrGetActionStatePose(*action, &getInfo, &state);
		}

		bool is_active()const override
		{
			return state.isActive;
		}

		bool changed_since_last_sync()const override
		{
			return true;
		}

		XrTime last_change_time()const override
		{
			return 0;
		}
		// pose is obtained via xrLocateSpace in space
		void printInfo()const override
		{
			printf("ActionState %s:\n", action->name.c_str());
			printf("\tis active: %s\n", state.isActive ? "True" : "False");
		}
	};
	template<>struct ActionState<ActionType::VibrationOutput> : ActionStateBase
	{
		using ret_type = XrPosef;
		static constexpr XrActionType _ActionType = XR_ACTION_TYPE_VIBRATION_OUTPUT;
		static constexpr XrStructureType _ActionStateType = XR_TYPE_ACTION_STATE_POSE;

		XrHapticVibration vibration;

		ActionState(Action* _action, Path* _subactionPath = nullptr)
			:
			ActionStateBase(_action, _subactionPath),
			vibration{ XR_TYPE_HAPTIC_VIBRATION, nullptr, XR_MIN_HAPTIC_DURATION, 0.5f, XR_FREQUENCY_UNSPECIFIED }
		{
		}

		void apply_haptic(float _amplitude = 0.5f,
			XrDuration _duration = XR_MIN_HAPTIC_DURATION,
			float _frequency = XR_FREQUENCY_UNSPECIFIED)
		{
			vibration = { XR_TYPE_HAPTIC_VIBRATION, nullptr, _duration, _frequency, _amplitude };

			XrHapticActionInfo hapticActionInfo{ XR_TYPE_HAPTIC_ACTION_INFO };
			hapticActionInfo.action = *action;
			if (subactionPath)
				hapticActionInfo.subactionPath = *subactionPath;
			xrApplyHapticFeedback(*action, &hapticActionInfo, (XrHapticBaseHeader*)&vibration);
		}

		void update_state() override {}
		// always true since this is not input action
		bool is_active()const override
		{
			return true;
		}

		bool changed_since_last_sync()const override
		{
			return true;
		}

		XrTime last_change_time()const override
		{
			return 0;
		}

		void stop_haptic()
		{
			XrHapticActionInfo hapticActionInfo{ XR_TYPE_HAPTIC_ACTION_INFO };
			hapticActionInfo.action = *action;
			if (subactionPath)
				hapticActionInfo.subactionPath = *subactionPath;
			// currently doesn't work on htc vive controller with steamvr
			xrStopHapticFeedback(*action, &hapticActionInfo);
		}

		void printInfo()const override
		{
			printf("ActionState %s:\n", action->name.c_str());
			printf("\tduration: %lld\n", vibration.duration);
			printf("\tfrequency: %f\n", vibration.frequency);
			printf("\tamplitude: %f\n", vibration.amplitude);
		}
	};

	struct SpaceLocation
	{
		XrResult res;
		XrSpaceLocation location;

		SpaceLocation()
			:
			res(XR_SUCCESS),
			location{ XR_TYPE_SPACE_LOCATION }
		{
		}

		bool valid()const
		{
			if (XR_UNQUALIFIED_SUCCESS(res) &&
				(location.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0 &&
				(location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) != 0)
				return true;
			return false;
		}
	};

	enum class SpaceType
	{
		Reference = 0,
		Action = 1,
		None = 2,
	};

	struct SpaceBase
	{
		XrSpace space;
		Session* session;

		SpaceBase(Session* _session)
			:
			space(XR_NULL_HANDLE),
			session(_session)
		{
		}

		~SpaceBase()
		{
			if (space != XR_NULL_HANDLE)
			{
				xrDestroySpace(space);
				space = XR_NULL_HANDLE;
				session = nullptr;
			}
		}

		operator bool()const
		{
			return space != XR_NULL_HANDLE;
		}

		operator XrSpace()const
		{
			return space;
		}

		operator XrSession()const
		{
			return session->session;
		}

		void locate_space(SpaceBase const& ref_space, XrTime _time, SpaceLocation* space_location)const
		{
			space_location->location = { XR_TYPE_SPACE_LOCATION };
			space_location->res = xrLocateSpace(space, ref_space, _time, &space_location->location);
		}
	};

	template<SpaceType spaceType = SpaceType::None>struct Space : SpaceBase {};

	template<>struct Space<SpaceType::Reference> : SpaceBase
	{
		Space(Session* _session) :SpaceBase(_session) {}
		Space(Session* _session, XrReferenceSpaceType reference_space_type)
			:
			SpaceBase(_session)
		{
			XrReferenceSpaceCreateInfo reference_space_create_info{ XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
			reference_space_create_info.poseInReferenceSpace.orientation.w = 1.f;
			reference_space_create_info.referenceSpaceType = reference_space_type;
			xrCreateReferenceSpace(*session, &reference_space_create_info, &space);
			if (space == XR_NULL_HANDLE)
			{
				printf("failed to create xr reference space!\n");
			}
		}
	};
	template<>struct Space<SpaceType::Action> :SpaceBase
	{
		Action* action;

		Space(Session* _session) :SpaceBase(_session), action(nullptr) {}
		Space(Session* _session, Action* _action, XrPath _subaction_path)
			:
			Space(_session)
		{
			bind_action(_action, _subaction_path);
		}

		void bind_action(Action* _action, XrPath _subaction_path)
		{
			if (space != XR_NULL_HANDLE)
			{
				xrDestroySpace(space);
				space = XR_NULL_HANDLE;
			}
			action = _action;
			XrActionSpaceCreateInfo action_space_create_info{ XR_TYPE_ACTION_SPACE_CREATE_INFO };
			action_space_create_info.action = *action;
			action_space_create_info.poseInActionSpace.orientation.w = 1.f;
			action_space_create_info.subactionPath = _subaction_path;
			xrCreateActionSpace(*session, &action_space_create_info, &space);
			if (space == XR_NULL_HANDLE)
			{
				printf("failed to create xr action space!\n");
			}
		}
	};



	enum class ControllerType
	{
		Simple = 0,
		GoogleDaydream = 1,
		HTCVive = 2,
		HTCVivePro = 3, // only one head
		MicrosoftMixedReality = 4,
		MicrosoftXbox = 5, // only one gamepad
		OculusGo = 6,
		OculusTouch = 7, // left right different
		ValveIndex = 8,
	};

	struct ControllerBase
	{
		Path profile;

		ControllerBase(Instance* _instance, char const* str = nullptr) :profile(_instance, str) {}
	};

	template<ControllerType controllerType = ControllerType::Simple>
	struct Controller : ControllerBase
	{
		Path select_click[2];
		Path menu_click[2];
		Path grip_pose[2];
		Path aim_pose[2];
		Path haptic[2];

		Controller(Instance* _instance)
			:
			ControllerBase(_instance, "/interaction_profiles/khr/simple_controller"),
			select_click{ _instance, _instance },
			menu_click{ _instance, _instance },
			grip_pose{ _instance, _instance },
			aim_pose{ _instance, _instance },
			haptic{ _instance, _instance }
		{
			std::string hands[2]{ "/user/hand/left", "/user/hand/right" };
			for (uint32_t hand(0); hand < 2; ++hand)
			{
				select_click[hand].create((hands[hand] + "/input/select/click").c_str());
				menu_click[hand].create((hands[hand] + "/input/menu/click").c_str());
				grip_pose[hand].create((hands[hand] + "/input/grip/pose").c_str());
				aim_pose[hand].create((hands[hand] + "/input/aim/pose").c_str());
				haptic[hand].create((hands[hand] + "/output/haptic").c_str());
			}
		}
	};

	template<>struct Controller<ControllerType::GoogleDaydream> : ControllerBase
	{
		Path select_click[2];
		Path trackpad_x[2];
		Path trackpad_y[2];
		Path trackpad_click[2];
		Path trackpad_touch[2];
		Path grip_pose[2];
		Path aim_pose[2];
		Path haptic[2];

		Controller(Instance* _instance)
			:
			ControllerBase(_instance, "/interaction_profiles/google/daydream_controller"),
			select_click{ _instance, _instance },
			trackpad_x{ _instance, _instance },
			trackpad_y{ _instance, _instance },
			trackpad_click{ _instance, _instance },
			trackpad_touch{ _instance, _instance },
			grip_pose{ _instance, _instance },
			aim_pose{ _instance, _instance },
			haptic{ _instance, _instance }
		{
			std::string hands[2]{ "/user/hand/left", "/user/hand/right" };
			for (uint32_t hand = 0; hand < 2; ++hand)
			{
				select_click[hand].create((hands[hand] + "/input/select/click").c_str());
				trackpad_x[hand].create((hands[hand] + "/input/trackpad/x").c_str());
				trackpad_y[hand].create((hands[hand] + "/input/trackpad/y").c_str());
				trackpad_click[hand].create((hands[hand] + "/input/trackpad/click").c_str());
				trackpad_touch[hand].create((hands[hand] + "/input/trackpad/touch").c_str());
				grip_pose[hand].create((hands[hand] + "/input/grip/pose").c_str());
				aim_pose[hand].create((hands[hand] + "/input/aim/pose").c_str());
				haptic[hand].create((hands[hand] + "/output/haptic").c_str());
			}
		}
	};
	template<>struct Controller<ControllerType::HTCVive> : ControllerBase
	{
		Path system_click[2];
		Path squeeze_click[2];
		Path menu_click[2];
		Path trigger_click[2];
		Path trigger_value[2];
		Path trackpad_x[2];
		Path trackpad_y[2];
		Path trackpad_click[2];
		Path trackpad_touch[2];
		Path grip_pose[2];
		Path aim_pose[2];
		Path haptic[2];

		Controller(Instance* _instance)
			:
			ControllerBase(_instance, "/interaction_profiles/htc/vive_controller"),
			system_click{ _instance, _instance },
			squeeze_click{ _instance, _instance },
			menu_click{ _instance, _instance },
			trigger_click{ _instance, _instance },
			trigger_value{ _instance, _instance },
			trackpad_x{ _instance, _instance },
			trackpad_y{ _instance, _instance },
			trackpad_click{ _instance, _instance },
			trackpad_touch{ _instance, _instance },
			grip_pose{ _instance, _instance },
			aim_pose{ _instance, _instance },
			haptic{ _instance, _instance }
		{
			std::string hands[2]{ "/user/hand/left", "/user/hand/right" };
			for (uint32_t hand = 0; hand < 2; ++hand)
			{
				system_click[hand].create((hands[hand] + "/input/system/click").c_str());
				squeeze_click[hand].create((hands[hand] + "/input/squeeze/click").c_str());
				menu_click[hand].create((hands[hand] + "/input/menu/click").c_str());
				trigger_click[hand].create((hands[hand] + "/input/trigger/click").c_str());
				trigger_value[hand].create((hands[hand] + "/input/trigger/value").c_str());
				trackpad_x[hand].create((hands[hand] + "/input/trackpad/x").c_str());
				trackpad_y[hand].create((hands[hand] + "/input/trackpad/y").c_str());
				trackpad_click[hand].create((hands[hand] + "/input/trackpad/click").c_str());
				trackpad_touch[hand].create((hands[hand] + "/input/trackpad/touch").c_str());
				grip_pose[hand].create((hands[hand] + "/input/grip/pose").c_str());
				aim_pose[hand].create((hands[hand] + "/input/aim/pose").c_str());
				haptic[hand].create((hands[hand] + "/output/haptic").c_str());
			}
		}
	};
	template<>struct Controller<ControllerType::HTCVivePro> : ControllerBase
	{
		Path system_click;
		Path volume_up_click;
		Path volume_down_click;
		Path mute_mic_click;

		Controller(Instance* _instance)
			:
			ControllerBase(_instance, "/interaction_profiles/htc/vive_pro"),
			system_click(_instance),
			volume_up_click(_instance),
			volume_down_click(_instance),
			mute_mic_click(_instance)
		{
			system_click.create("/user/head/input/system/click");
			volume_up_click.create("/user/head/input/volume_up/click");
			volume_down_click.create("/user/head/input/volume_down/click");
			mute_mic_click.create("/user/head/input/mute_mic/click");
		}
	};
	template<>struct Controller<ControllerType::MicrosoftMixedReality> : ControllerBase
	{
		Path menu_click[2];
		Path squeeze_click[2];
		Path trigger_value[2];
		Path thumbstick_x[2];
		Path thumbstick_y[2];
		Path thumbstick_click[2];
		Path trackpad_x[2];
		Path trackpad_y[2];
		Path trackpad_click[2];
		Path trackpad_touch[2];
		Path grip_pose[2];
		Path aim_pose[2];
		Path haptic[2];

		Controller(Instance* _instance)
			:
			ControllerBase(_instance, "/interaction_profiles/microsoft/motion_controller"),
			menu_click{ _instance, _instance },
			squeeze_click{ _instance, _instance },
			trigger_value{ _instance, _instance },
			thumbstick_x{ _instance, _instance },
			thumbstick_y{ _instance, _instance },
			thumbstick_click{ _instance, _instance },
			trackpad_x{ _instance, _instance },
			trackpad_y{ _instance, _instance },
			trackpad_click{ _instance, _instance },
			trackpad_touch{ _instance, _instance },
			grip_pose{ _instance, _instance },
			aim_pose{ _instance, _instance },
			haptic{ _instance, _instance }
		{
			std::string hands[2]{ "/user/hand/left", "/user/hand/right" };
			for (uint32_t hand = 0; hand < 2; ++hand)
			{
				menu_click[hand].create((hands[hand] + "/input/menu/click").c_str());
				squeeze_click[hand].create((hands[hand] + "/input/squeeze/click").c_str());
				trigger_value[hand].create((hands[hand] + "/input/trigger/value").c_str());
				thumbstick_x[hand].create((hands[hand] + "/input/thumbstick/x").c_str());
				thumbstick_y[hand].create((hands[hand] + "/input/thumbstick/y").c_str());
				thumbstick_click[hand].create((hands[hand] + "/input/thumbstick/click").c_str());
				trackpad_x[hand].create((hands[hand] + "/input/trackpad/x").c_str());
				trackpad_y[hand].create((hands[hand] + "/input/trackpad/y").c_str());
				trackpad_click[hand].create((hands[hand] + "/input/trackpad/click").c_str());
				trackpad_touch[hand].create((hands[hand] + "/input/trackpad/touch").c_str());
				grip_pose[hand].create((hands[hand] + "/input/grip/pose").c_str());
				aim_pose[hand].create((hands[hand] + "/input/aim/pose").c_str());
				haptic[hand].create((hands[hand] + "/output/haptic").c_str());
			}
		}
	};
	template<>struct Controller<ControllerType::MicrosoftXbox> : ControllerBase
	{
		Path menu_click;
		Path view_click;
		Path a_click;
		Path b_click;
		Path x_click;
		Path y_click;
		Path dpad_down_click;
		Path dpad_right_click;
		Path dpad_up_click;
		Path dpad_left_click;
		Path shoulder_left_click;
		Path shoulder_right_click;
		Path thumbstick_left_click;
		Path thumbstick_right_click;
		Path trigger_left_value;
		Path trigger_right_value;
		Path thumbstick_left_x;
		Path thumbstick_left_y;
		Path thumbstick_right_x;
		Path thumbstick_right_y;
		Path haptic_left;
		Path haptic_right;
		Path haptic_left_trigger;
		Path haptic_right_trigger;

		Controller(Instance* _instance)
			:
			ControllerBase(_instance, "/interaction_profiles/microsoft/xbox_controller"),
			menu_click(_instance),
			view_click(_instance),
			a_click(_instance),
			b_click(_instance),
			x_click(_instance),
			y_click(_instance),
			dpad_down_click(_instance),
			dpad_right_click(_instance),
			dpad_up_click(_instance),
			dpad_left_click(_instance),
			shoulder_left_click(_instance),
			shoulder_right_click(_instance),
			thumbstick_left_click(_instance),
			thumbstick_right_click(_instance),
			trigger_left_value(_instance),
			trigger_right_value(_instance),
			thumbstick_left_x(_instance),
			thumbstick_left_y(_instance),
			thumbstick_right_x(_instance),
			thumbstick_right_y(_instance),
			haptic_left(_instance),
			haptic_right(_instance),
			haptic_left_trigger(_instance),
			haptic_right_trigger(_instance)
		{
			menu_click.create("/user/gamepad/input/menu/click");
			view_click.create("/user/gamepad/input/view/click");
			a_click.create("/user/gamepad/input/a/click");
			b_click.create("/user/gamepad/input/b/click");
			x_click.create("/user/gamepad/input/x/click");
			y_click.create("/user/gamepad/input/y/click");
			dpad_down_click.create("/user/gamepad/input/dpad_down/click");
			dpad_right_click.create("/user/gamepad/input/dpad_right/click");
			dpad_up_click.create("/user/gamepad/input/dpad_up/click");
			dpad_left_click.create("/user/gamepad/input/dpad_left/click");
			shoulder_left_click.create("/user/gamepad/input/shoulder_left/click");
			shoulder_right_click.create("/user/gamepad/input/shoulder_right/click");
			thumbstick_left_click.create("/user/gamepad/input/thumbstick_left/click");
			thumbstick_right_click.create("/user/gamepad/input/thumbstick_right/click");
			trigger_left_value.create("/user/gamepad/input/trigger_left/value");
			trigger_right_value.create("/user/gamepad/input/trigger_right/value");
			thumbstick_left_x.create("/user/gamepad/input/thumbstick_left/x");
			thumbstick_left_y.create("/user/gamepad/input/thumbstick_left/y");
			thumbstick_right_x.create("/user/gamepad/input/thumbstick_right/x");
			thumbstick_right_y.create("/user/gamepad/input/thumbstick_right/y");
			haptic_left.create("/user/gamepad/output/haptic_left");
			haptic_right.create("/user/gamepad/output/haptic_right");
			haptic_left_trigger.create("/user/gamepad/output/haptic_left_trigger");
			haptic_right_trigger.create("/user/gamepad/output/haptic_right_trigger");
		}
	};
	template<>struct Controller<ControllerType::OculusGo> : ControllerBase
	{
		Path system_click[2];
		Path trigger_click[2];
		Path back_click[2];
		Path trackpad_x[2];
		Path trackpad_y[2];
		Path trackpad_click[2];
		Path trackpad_touch[2];
		Path grip_pose[2];
		Path aim_pose[2];

		Controller(Instance* _instance)
			:
			ControllerBase(_instance, "/interaction_profiles/oculus/go_controller"),
			system_click{ _instance, _instance },
			trigger_click{ _instance, _instance },
			back_click{ _instance, _instance },
			trackpad_x{ _instance, _instance },
			trackpad_y{ _instance, _instance },
			trackpad_click{ _instance, _instance },
			trackpad_touch{ _instance, _instance },
			grip_pose{ _instance, _instance },
			aim_pose{ _instance, _instance }
		{
			std::string hands[2]{ "/user/hand/left", "/user/hand/right" };
			for (uint32_t hand = 0; hand < 2; ++hand)
			{
				system_click[hand].create((hands[hand] + "/input/system/click").c_str());
				trigger_click[hand].create((hands[hand] + "/input/trigger/click").c_str());
				back_click[hand].create((hands[hand] + "/input/back/click").c_str());
				trackpad_x[hand].create((hands[hand] + "/input/trackpad/x").c_str());
				trackpad_y[hand].create((hands[hand] + "/input/trackpad/y").c_str());
				trackpad_click[hand].create((hands[hand] + "/input/trackpad/click").c_str());
				trackpad_touch[hand].create((hands[hand] + "/input/trackpad/touch").c_str());
				grip_pose[hand].create((hands[hand] + "/input/grip/pose").c_str());
				aim_pose[hand].create((hands[hand] + "/input/aim/pose").c_str());
			}
		}
	};
	template<>struct Controller<ControllerType::OculusTouch> : ControllerBase
	{
		// left
		Path x_click;
		Path x_touch;
		Path y_click;
		Path y_touch;
		Path menu_click;
		// right
		Path a_click;
		Path a_touch;
		Path b_click;
		Path b_touch;
		Path system_click;
		// common
		Path squeeze_value[2];
		Path trigger_value[2];
		Path trigger_touch[2];
		Path thumbstick_x[2];
		Path thumbstick_y[2];
		Path thumbstick_click[2];
		Path thumbstick_touch[2];
		Path thumbrest_touch[2];
		Path grip_pose[2];
		Path aim_pose[2];
		Path haptic[2];

		Controller(Instance* _instance)
			:
			ControllerBase(_instance, "/interaction_profiles/oculus/touch_controller"),
			x_click(_instance),
			x_touch(_instance),
			y_click(_instance),
			y_touch(_instance),
			menu_click(_instance),
			a_click(_instance),
			a_touch(_instance),
			b_click(_instance),
			b_touch(_instance),
			system_click(_instance),
			squeeze_value{ _instance, _instance },
			trigger_value{ _instance, _instance },
			trigger_touch{ _instance, _instance },
			thumbstick_x{ _instance, _instance },
			thumbstick_y{ _instance, _instance },
			thumbstick_click{ _instance, _instance },
			thumbstick_touch{ _instance, _instance },
			thumbrest_touch{ _instance, _instance },
			grip_pose{ _instance, _instance },
			aim_pose{ _instance, _instance },
			haptic{ _instance, _instance }
		{
			x_click.create("/user/hand/left/input/x/click");
			x_touch.create("/user/hand/left/input/x/touch");
			y_click.create("/user/hand/left/input/y/click");
			y_touch.create("/user/hand/left/input/y/touch");
			menu_click.create("/user/hand/left/input/menu/click");

			a_click.create("/user/hand/right/input/a/click");
			a_touch.create("/user/hand/right/input/a/touch");
			b_click.create("/user/hand/right/input/b/click");
			b_touch.create("/user/hand/right/input/b/touch");
			system_click.create("/user/hand/right/input/system/click");

			std::string hands[2]{ "/user/hand/left", "/user/hand/right" };
			for (uint32_t hand = 0; hand < 2; ++hand)
			{
				squeeze_value[hand].create((hands[hand] + "/input/squeeze/value").c_str());
				trigger_value[hand].create((hands[hand] + "/input/trigger/value").c_str());
				trigger_touch[hand].create((hands[hand] + "/input/trigger/touch").c_str());
				thumbstick_x[hand].create((hands[hand] + "/input/thumbstick/x").c_str());
				thumbstick_y[hand].create((hands[hand] + "/input/thumbstick/y").c_str());
				thumbstick_click[hand].create((hands[hand] + "/input/thumbstick/click").c_str());
				thumbstick_touch[hand].create((hands[hand] + "/input/thumbstick/touch").c_str());
				thumbrest_touch[hand].create((hands[hand] + "/input/thumbrest/touch").c_str());
				grip_pose[hand].create((hands[hand] + "/input/grip/pose").c_str());
				aim_pose[hand].create((hands[hand] + "/input/aim/pose").c_str());
				haptic[hand].create((hands[hand] + "/output/haptic").c_str());
			}
		}
	};
	template<>struct Controller<ControllerType::ValveIndex> : ControllerBase
	{
		Path system_click[2];
		Path system_touch[2];
		Path a_click[2];
		Path a_touch[2];
		Path b_click[2];
		Path b_touch[2];
		Path squeeze_value[2];
		Path squeeze_force[2];
		Path trigger_click[2];
		Path trigger_value[2];
		Path trigger_touch[2];
		Path thumbstick_x[2];
		Path thumbstick_y[2];
		Path thumbstick_click[2];
		Path thumbstick_touch[2];
		Path trackpad_x[2];
		Path trackpad_y[2];
		Path trackpad_force[2];
		Path trackpad_touch[2];
		Path grip_pose[2];
		Path aim_pose[2];
		Path haptic[2];

		Controller(Instance* _instance)
			:
			ControllerBase(_instance, "/interaction_profiles/valve/index_controller"),
			system_click{ _instance, _instance },
			system_touch{ _instance, _instance },
			a_click{ _instance, _instance },
			a_touch{ _instance, _instance },
			b_click{ _instance, _instance },
			b_touch{ _instance, _instance },
			squeeze_value{ _instance, _instance },
			squeeze_force{ _instance, _instance },
			trigger_click{ _instance, _instance },
			trigger_value{ _instance, _instance },
			trigger_touch{ _instance, _instance },
			thumbstick_x{ _instance, _instance },
			thumbstick_y{ _instance, _instance },
			thumbstick_click{ _instance, _instance },
			thumbstick_touch{ _instance, _instance },
			trackpad_x{ _instance, _instance },
			trackpad_y{ _instance, _instance },
			trackpad_force{ _instance, _instance },
			trackpad_touch{ _instance, _instance },
			grip_pose{ _instance, _instance },
			aim_pose{ _instance, _instance },
			haptic{ _instance, _instance }
		{
			std::string hands[2]{ "/user/hand/left", "/user/hand/right" };
			for (uint32_t hand = 0; hand < 2; ++hand)
			{
				system_click[hand].create((hands[hand] + "/input/system/click").c_str());
				system_touch[hand].create((hands[hand] + "/input/system/touch").c_str());
				a_click[hand].create((hands[hand] + "/input/a/click").c_str());
				a_touch[hand].create((hands[hand] + "/input/a/touch").c_str());
				b_click[hand].create((hands[hand] + "/input/b/click").c_str());
				b_touch[hand].create((hands[hand] + "/input/b/touch").c_str());
				squeeze_value[hand].create((hands[hand] + "/input/squeeze/value").c_str());
				squeeze_force[hand].create((hands[hand] + "/input/squeeze/force").c_str());
				trigger_click[hand].create((hands[hand] + "/input/trigger/click").c_str());
				trigger_value[hand].create((hands[hand] + "/input/trigger/value").c_str());
				trigger_touch[hand].create((hands[hand] + "/input/trigger/touch").c_str());
				thumbstick_x[hand].create((hands[hand] + "/input/thumbstick/x").c_str());
				thumbstick_y[hand].create((hands[hand] + "/input/thumbstick/y").c_str());
				thumbstick_click[hand].create((hands[hand] + "/input/thumbstick/click").c_str());
				thumbstick_touch[hand].create((hands[hand] + "/input/thumbstick/touch").c_str());
				trackpad_x[hand].create((hands[hand] + "/input/trackpad/x").c_str());
				trackpad_y[hand].create((hands[hand] + "/input/trackpad/y").c_str());
				trackpad_force[hand].create((hands[hand] + "/input/trackpad/force").c_str());
				trackpad_touch[hand].create((hands[hand] + "/input/trackpad/touch").c_str());
				grip_pose[hand].create((hands[hand] + "/input/grip/pose").c_str());
				aim_pose[hand].create((hands[hand] + "/input/aim/pose").c_str());
				haptic[hand].create((hands[hand] + "/output/haptic").c_str());
			}
		}
	};

	template<ControllerType controllerType>
	void bind_interaction_profile(Controller<controllerType> const& controller, std::vector<XrActionSuggestedBinding> const& bindings)
	{
		XrInteractionProfileSuggestedBinding suggestedBindings{ XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING };
		suggestedBindings.interactionProfile = controller.profile.path;
		suggestedBindings.suggestedBindings = bindings.data();
		suggestedBindings.countSuggestedBindings = (uint32_t)bindings.size();
		xrSuggestInteractionProfileBindings(controller.profile.instance->instance, &suggestedBindings);
	}

	struct ControllerSet
	{
		Controller<ControllerType::Simple>					simple_controller;
		Controller<ControllerType::GoogleDaydream>			google_controller;
		Controller<ControllerType::HTCVive>					vive_controller;
		Controller<ControllerType::HTCVivePro>				vive_pro_controller;
		Controller<ControllerType::MicrosoftMixedReality>	msft_mr_controller;
		Controller<ControllerType::MicrosoftXbox>			msft_xbox_controller;
		Controller<ControllerType::OculusGo>				oculus_go_controller;
		Controller<ControllerType::OculusTouch>				oculus_touch_controller;
		Controller<ControllerType::ValveIndex>				valve_index_controller;

		ControllerSet(Instance* _instance)
			:
			simple_controller(_instance),
			google_controller(_instance),
			vive_controller(_instance),
			vive_pro_controller(_instance),
			msft_mr_controller(_instance),
			msft_xbox_controller(_instance),
			oculus_go_controller(_instance),
			oculus_touch_controller(_instance),
			valve_index_controller(_instance)
		{
		}
	};

	// if farZ is negative, set far plane to infinity
	Math::mat4<float> get_projection(XrFovf const& fov, float nearZ, float farZ = -1.f);

	// if not inv, get the transformation matrix from reference space to target space for the XrPosef
	Math::mat4<float> get_transform(XrPosef pose, bool inv);

	// interface for a OpenXR app that would attach to an OpenGL app
	struct XrOpenGL
	{
		virtual float nearZ(uint32_t eye) = 0;
		virtual float farZ(uint32_t eye) = 0;

		virtual void xrinit() {}
		virtual void pullActions() {} // before rendering
		virtual void update() {} // after rendering both eyes

		virtual void setViewport(XrRect2Di const& viewport) = 0; // set before rendering one eye
		virtual void setViewMat(Math::mat4<float>const& app2view) = 0; // set before rendering one eye
		virtual void setProjMat(Math::mat4<float>const& view2proj) = 0; // set before rendering one eye

		virtual void xrRender(uint32_t eye) {} // render one eye
	};

	// single OpenXR app attached to an OpenGL app, independent of XrOS,
	// has its own framebuffer, inherit from this to finish an app
	// you need to write your own action codes
	struct XrRunner
	{
		Instance instance;
		System system;
		Session session;
		Frame frame;
		Views views;
		EventPoller poller;
		ActionSet actionSet;
		ControllerSet controllerSet;

		Space<SpaceType::Reference> appSpace;
		Space<SpaceType::Reference> viewSpace;

		bool sessionRunning;
		bool exitRenderLoop;
		bool requestRestart;

		// todo: create validViewCount framebuffers
		std::vector<OpenGL::FrameBufferSource*> colors;
		std::vector<OpenGL::FrameBufferSource*> depths;
		std::vector<OpenGL::FrameBuffer> framebuffers;

		XrOpenGL* xrOpenGL;

		XrRunner(GLFWwindow* _window, std::vector<char const*> extraExtensions = {}, bool _enable_depth = false)
			:
			instance("XrRunner", extraExtensions, _enable_depth),
			system(&instance),
			session(&system, _window),
			frame(&session, XR_ENVIRONMENT_BLEND_MODE_OPAQUE),
			views(&session, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO),
			poller(&instance),
			actionSet(&instance, "action_set_default", "Action Set Default", 0),
			controllerSet(&instance),
			appSpace(&session, XR_REFERENCE_SPACE_TYPE_LOCAL),
			viewSpace(&session, XR_REFERENCE_SPACE_TYPE_VIEW),
			sessionRunning(false),
			exitRenderLoop(false),
			requestRestart(false),
			xrOpenGL(nullptr)
		{
			instance.printInfo();
			session.printSystemInfo();
			session.printReferenceSpace();
			views.printViewConfig();
			views.printViewInfo();
			views.printSwapchainInfo();
		}

		~XrRunner()
		{
			for (auto color : colors)delete color;
			for (auto depth : depths)delete depth;
		}

		virtual void bind_controllers() = 0;
		virtual void update_actions() = 0;
		virtual void before_render(XrTime predictedDisplayTime) {}

		void handle_session_state_event(XrEventDataSessionStateChanged const& stateChangedEvent)
		{
			const XrSessionState oldState = session.state;
			session.state = stateChangedEvent.state;
			printf("XrEventDataSessionStateChanged: state %d->%d session=%lld time=%lld\n",
				oldState, session.state, stateChangedEvent.session, stateChangedEvent.time);

			if ((stateChangedEvent.session != XR_NULL_HANDLE) && (stateChangedEvent.session != session))
			{
				printf("XrEventDataSessionStateChanged for unknown session!\n");
				return;
			}
			switch (session.state)
			{
			case XR_SESSION_STATE_READY:
			{
				XrSessionBeginInfo sessionBeginInfo{ XR_TYPE_SESSION_BEGIN_INFO };
				sessionBeginInfo.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
				xrBeginSession(session, &sessionBeginInfo);
				sessionRunning = true;
				break;
			}
			case XR_SESSION_STATE_STOPPING:
			{
				sessionRunning = false;
				xrEndSession(session);
				break;
			}
			case XR_SESSION_STATE_EXITING:
			{
				// Do not attempt to restart because user closed this session.
				exitRenderLoop = true;
				requestRestart = false;
				break;
			}
			case XR_SESSION_STATE_LOSS_PENDING:
			{
				// Poll for a new instance.
				exitRenderLoop = true;
				requestRestart = true;
				break;
			}
			default:break;
			}
		}

		void poll_events()
		{
			exitRenderLoop = false;
			requestRestart = false;
			while (const XrEventDataBaseHeader* event = poller.poll())
			{
				switch (event->type)
				{
				case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
				{
					const auto& instanceLossPending = *reinterpret_cast<const XrEventDataInstanceLossPending*>(event);
					printf("XrEventDataInstanceLossPending by %lld", instanceLossPending.lossTime);

					exitRenderLoop = true;
					requestRestart = true;
					return;
				}
				case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED:
				{
					auto sessionStateChangedEvent = *reinterpret_cast<const XrEventDataSessionStateChanged*>(event);
					handle_session_state_event(sessionStateChangedEvent);
					break;
				}
				case XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED:
				{
					// 	log_action_source_name(grabAction, "Grab");
					// 	log_action_source_name(quitAction, "Quit");
					// 	log_action_source_name(poseAction, "Pose");
					// 	log_action_source_name(vibrateAction, "Vibrate");
					break;
				}
				case XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING:
				default:
				{
					printf("Ignoring event type %d", event->type);
					break;
				}
				}
			}
		}

		void bindXrOpenGL(XrOpenGL* _xrOpenGL)
		{
			xrOpenGL = _xrOpenGL;
			xrOpenGL->xrinit();
		}

		bool render_layer(XrTime predictedDisplayTime, std::vector<XrCompositionLayerProjectionView>& colorViews, std::vector<XrCompositionLayerDepthInfoKHR>& depthViews, XrCompositionLayerProjection& layer)
		{
			if (!views.locate(predictedDisplayTime, appSpace))
				return false;

			if (framebuffers.size() != views.validViewCount)
			{
				colors.resize(views.validViewCount);
				depths.resize(views.validViewCount);
				framebuffers.resize(views.validViewCount);
				for (int i(0); i < views.validViewCount; ++i)
				{
					colors[i] = nullptr;
					depths[i] = nullptr;
					framebuffers[i].create();
				}
			}

			before_render(predictedDisplayTime);
			xrOpenGL->pullActions();

			bool enabledDepth(instance.enabledDepth);
			colorViews.resize(views.validViewCount);
			if (enabledDepth)
				depthViews.resize(views.validViewCount);

			// Render view to the appropriate part of the swapchain image.
			for (uint32_t i = 0; i < views.validViewCount; i++)
			{
				// Each view has a separate swapchain which is acquired, rendered to, and released.

				SwapchainImages& colorSwapchain = views.colorSwapchainImages[i];

				uint32_t colorSwapchainImageIndex(0);
				uint32_t depthSwapchainImageIndex(0);
				colorSwapchain.acquire(&colorSwapchainImageIndex);
				colorSwapchain.wait();

				colorViews[i] = { XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW };
				colorViews[i].pose = views.views[i].pose;
				colorViews[i].fov = views.views[i].fov;
				colorViews[i].subImage.swapchain = colorSwapchain;
				// colorViews[i].subImage.imageArrayIndex = i; // 0 for ogl
				colorViews[i].subImage.imageRect.offset = { 0, 0 };
				colorViews[i].subImage.imageRect.extent = { colorSwapchain.width, colorSwapchain.height };


				// prepare frame buffer
				using namespace OpenGL;
				GLuint color_texture(0), depth_texture(0);
				float nearZ = xrOpenGL->nearZ(i);
				float farZ = xrOpenGL->farZ(i);
				if (!colors[i])
				{
					colors[i] = new FrameBufferSourceTexture(FrameBufferAttachment::Color0, Texture2D);
					framebuffers[i].color = colors[i];
				}
				color_texture = colorSwapchain.swapchainImages[colorSwapchainImageIndex].image;

				if (enabledDepth)
				{
					SwapchainImages& depthSwapchain = views.depthSwapchainImages[i];
					depthSwapchain.acquire(&depthSwapchainImageIndex);
					depthSwapchain.wait();
					depthViews[i] = { XR_TYPE_COMPOSITION_LAYER_DEPTH_INFO_KHR };
					depthViews[i].subImage.swapchain = depthSwapchain;
					// depthViews[i].subImage.imageArrayIndex = i; // 0 for ogl
					depthViews[i].subImage.imageRect.offset = { 0, 0 };
					depthViews[i].subImage.imageRect.extent = { depthSwapchain.width, depthSwapchain.height };
					depthViews[i].minDepth = 0.f;
					depthViews[i].maxDepth = 1.f;
					depthViews[i].nearZ = nearZ;
					depthViews[i].farZ = farZ;
					colorViews[i].next = &depthViews[i];
					depth_texture = depthSwapchain.swapchainImages[depthSwapchainImageIndex].image;
					if (depths[i])
					{
						if (!depths[i]->is_texture())
						{
							delete depths[i];
							depths[i] = nullptr;
						}
					}
					if (!depths[i])
					{
						depths[i] = new FrameBufferSourceTexture(FrameBufferAttachment::Depth, Texture2D);
						framebuffers[i].depth = depths[i];
					}
				}
				else
				{
					if (!depths[i])
					{
						depths[i] = new FrameBufferSourceRenderBuffer(FrameBufferAttachment::Depth, RenderBufferFormat::Depth32F);
						framebuffers[i].depth = depths[i];
					}
				}
				framebuffers[i].set_texture(color_texture, depth_texture);
			}

			for (uint32_t i = 0; i < views.validViewCount; i++)
			{
				framebuffers[i].bind();
				xrOpenGL->setProjMat(get_projection(views.views[i].fov, xrOpenGL->nearZ(i), xrOpenGL->farZ(i)));
				xrOpenGL->setViewMat(get_transform(views.views[i].pose, true));
				xrOpenGL->setViewport(colorViews[i].subImage.imageRect);
				xrOpenGL->xrRender(i);
				framebuffers[i].unbind();
			}

			for (uint32_t i = 0; i < views.validViewCount; i++)
			{
				views.colorSwapchainImages[i].release();
				if (enabledDepth)
					views.depthSwapchainImages[i].release();
			}

			xrOpenGL->update();

			layer.space = appSpace;
			layer.layerFlags = 0;
			layer.viewCount = (uint32_t)colorViews.size();
			layer.views = colorViews.data();
			return true;
		}

		void render()
		{
			frame.wait();
			frame.begin();

			std::vector<XrCompositionLayerBaseHeader*> layers;
			std::vector<XrCompositionLayerProjectionView> projectionLayerViews;
			std::vector<XrCompositionLayerDepthInfoKHR> projectionLayerDepths;
			XrCompositionLayerProjection layer{ XR_TYPE_COMPOSITION_LAYER_PROJECTION };

			if (frame.should_render())
				if (render_layer(frame.predicted_display_time(), projectionLayerViews, projectionLayerDepths, layer))
					layers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(&layer));

			frame.end(layers);
		}

		bool update()
		{
			poll_events();
			if (exitRenderLoop)
				return false;
			update_actions();
			render();
			return true;
		}
	};
}