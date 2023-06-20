#include <_OpenXR.h>
#include <_ImGui.h>
#include <_NBody.h>
#ifdef _CUDA
using NBodyImpl = OpenGL::NBodyCUDAImpl;
#else
using NBodyImpl = OpenGL::NBodyOpenGLImpl;
#endif
#include <array>
#include <map>

inline void printXrVersion(XrVersion ver)
{
	printf("%d.%d.%d", XR_VERSION_MAJOR(ver), XR_VERSION_MINOR(ver), XR_VERSION_PATCH(ver));
}

struct Cube
{
	XrPosef Pose;
	XrVector3f Scale;
};

namespace OpenGL
{
	struct VRRenderer : Program
	{
		BufferConfig* particlesArray;
		Buffer transBuffer;
		BufferConfig transUniform;
		VertexAttrib positions;
		VertexAttrib velocities;

		VRRenderer(SourceManager* _sm, BufferConfig* _particlesArray, Transform::BufferData* _trans)
			:
			Program(_sm, "Renderer", Vector< VertexAttrib*>{&positions, & velocities}),
			particlesArray(_particlesArray),
			transBuffer(_trans),
			transUniform(&transBuffer, UniformBuffer, 0),
			positions(_particlesArray, 0, VertexAttrib::three, VertexAttrib::Float, false, sizeof(NBodyData::Particles::Particle), 0, 0),
			velocities(_particlesArray, 1, VertexAttrib::three, VertexAttrib::Float, false, sizeof(NBodyData::Particles::Particle), 16, 0)
		{
			init();
		}
		virtual void initBufferData()override
		{
		}
		virtual void run()override
		{
			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			glClearDepth(1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			transUniform.refreshData();
			glDrawArrays(GL_POINTS, 0, particlesArray->buffer->data->size() / sizeof(NBodyData::Particles::Particle));
		}
	};

	struct HelloVR_GL :OpenGL
	{
		uint32_t width, height;
		GLuint colorTexture;
		GLuint depthBuffer;
		GLuint frameBuffer;

		XrRect2Di viewport;
		XrFovf fov;
		XrPosef view2appPose;
		XrPosef hand2appPose;
		float handScale;

		SourceManager sm;
		NBodyData nbodyData;
		NBodyImpl nbody;
		Transform::BufferData trans;
		VRRenderer renderer;

		HelloVR_GL(unsigned int _blocks, bool _experiment)
			:
			width(0),
			height(0),
			colorTexture(0),
			depthBuffer(0),
			frameBuffer(0),
			viewport{ { 0,0 }, { 0,0 } },
			fov{ 0 },
			view2appPose{ 0 },
			hand2appPose{ 0 },
			handScale(0.03f),

			sm("./"),
			nbodyData(_blocks, _experiment, &sm),
			nbody(&nbodyData, &sm),
			renderer(&sm, &nbodyData.particlesArray, &trans)
		{

		}

		~HelloVR_GL()
		{
			glDeleteFramebuffers(1, &frameBuffer);
		}

		void get_projection(Math::mat4<float>* result, const float nearZ = 0.05f, const float farZ = 100.f)
		{
			const float tanLeft = tanf(fov.angleLeft);
			const float tanRight = tanf(fov.angleRight);
			const float tanUp = tanf(fov.angleUp);
			float const tanDown = tanf(fov.angleDown);

			const float width = tanRight - tanLeft;
			const float height = tanUp - tanDown;
			const float offsetZ = nearZ;

			*result = 0.f;
			result->array[0][0] = 2.0f / width;
			result->array[0][2] = (tanRight + tanLeft) / width;
			result->array[1][1] = 2.0f / height;
			result->array[1][2] = (tanUp + tanDown) / height;
			result->array[3][2] = -1.0f;

			if (farZ <= nearZ)
			{
				// place the far plane at infinity
				result->array[2][2] = -1.0f;
				result->array[2][3] = -(nearZ + offsetZ);
			}
			else
			{
				// normaarray projection
				result->array[2][2] = -(farZ + offsetZ) / (farZ - nearZ);
				result->array[2][3] = -(farZ * (nearZ + offsetZ)) / (farZ - nearZ);
			}
		}

		void get_pose(Math::mat4<float>* result, XrPosef pose, bool inv)
		{
			XrQuaternionf& quat = pose.orientation;
			XrVector3f& pos = pose.position;

			// inverse
			if (inv)
			{
				quat.x = -quat.x;
				quat.y = -quat.y;
				quat.z = -quat.z;
			}

			*result = 0;
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

			result->array[0][0] = 1.0f - yy2 - zz2;
			result->array[0][1] = xy2 - wz2;
			result->array[0][2] = xz2 + wy2;

			result->array[1][0] = xy2 + wz2;
			result->array[1][1] = 1.0f - xx2 - zz2;
			result->array[1][2] = yz2 - wx2;

			result->array[2][0] = xz2 - wy2;
			result->array[2][1] = yz2 + wx2;
			result->array[2][2] = 1.0f - xx2 - yy2;

			Math::vec3<float> pos_vec{ pos.x, pos.y, pos.z};
			if (inv)
			{
				result->setCol((*result, -pos_vec), 3);
			}
			else
			{
				result->setCol(pos_vec, 3);
			}

			result->array[3][3] = 1.0f;
		}

		void get_scale(Math::mat4<float>* result, float scale)
		{
			*result = 0;
			result->array[0][0] = scale;
			result->array[1][1] = scale;
			result->array[2][2] = scale;
			result->array[3][3] = 1.f;
		}

		void bind_framebuffer()
		{
			GLint _width(0), _height(0);
			glBindTexture(GL_TEXTURE_2D, colorTexture);
			glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &_width);
			glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &_height);

			glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);

			if (width != _width || height != _height)
			{
				width = _width;
				height = _height;
				printf("Got texture size: [%d, %d]\n", width, height);

				if (depthBuffer) glDeleteRenderbuffers(1, &depthBuffer);

				glGenRenderbuffers(1, &depthBuffer);
				glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
				glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, width, height);
				glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);
			}
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);
		}

		void unbind_framebuffer()
		{
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
		}

		virtual void init(FrameScale const& _size)override
		{
			glGenFramebuffers(1, &frameBuffer);
			glPointSize(1);
			glEnable(GL_DEPTH_TEST);
			renderer.transUniform.dataInit();
			renderer.particlesArray->dataInit();
			nbody.init();
		}

		virtual void run()override
		{
			if (colorTexture)
			{
				bind_framebuffer();

				glViewport(viewport.offset.x, viewport.offset.y, viewport.extent.width, viewport.extent.height);
				Math::mat4<float>proj, rot_app2view, scale, rot_hand2app;
				get_projection(&proj);
				get_pose(&rot_app2view, view2appPose, true);
				get_pose(&rot_hand2app, hand2appPose, false);
				get_scale(&scale, handScale);
				trans.ans = (proj, (rot_app2view, (rot_hand2app, scale)));

				renderer.use();
				renderer.run();

				unbind_framebuffer();
				colorTexture = 0;
			}
			glClearColor(0.0f, 0.5f, 0.0f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT);
		}

		void update_phy()
		{
			nbody.run();
		}
	};
}

namespace OpenXR
{
	struct GalaxyInHand
	{
		Instance instance;
		System system;
		Session session;
		Frame frame;
		Views views;
		EventPoller poller;
		ActionSet actionSet;

		std::vector<Path> handSubactionPath;

		Action grabAction;
		Action poseAction;
		Action vibrateAction;
		Action quitAction;

		ActionState<ActionType::FloatInput> grabActionState[2];
		ActionState<ActionType::PoseInput> poseActionState[2];
		ActionState<ActionType::VibrationOutput> vibrateActionState[2];
		ActionState<ActionType::BoolInput> quitActionState;

		Space<SpaceType::Reference> appSpace;
		Space<SpaceType::Reference> viewSpace;
		Space<SpaceType::Action> handSpace[2];

		float handScale[2];
		XrBool32 handActive[2];

		bool sessionRunning;
		bool exitRenderLoop;
		bool requestRestart;

		OpenGL::HelloVR_GL* app;

		GalaxyInHand(GLFWwindow* _window, OpenGL::HelloVR_GL* _app)
			:
			instance("GalaxyInHand"),
			system(&instance),
			session(&system, _window),
			frame(&session, XR_ENVIRONMENT_BLEND_MODE_OPAQUE),
			views(&session, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO),
			poller(&instance),
			actionSet(&instance, "gameplay", "Gameplay", 0),

			handSubactionPath{ {&instance, "/user/hand/left"}, {&instance, "/user/hand/right"} },

			grabAction(&actionSet, "grab_object", "Grab Object", XR_ACTION_TYPE_FLOAT_INPUT, handSubactionPath),
			poseAction(&actionSet, "hand_pose", "Hand Pose", XR_ACTION_TYPE_POSE_INPUT, handSubactionPath),
			vibrateAction(&actionSet, "vibrate_hand", "Vibrate Hand", XR_ACTION_TYPE_VIBRATION_OUTPUT, handSubactionPath),
			quitAction(&actionSet, "quit_session", "Quit Session", XR_ACTION_TYPE_BOOLEAN_INPUT),

			grabActionState{ {&grabAction, &handSubactionPath[0]}, {&grabAction, &handSubactionPath[1]} },
			poseActionState{ {&poseAction, &handSubactionPath[0]}, {&poseAction, &handSubactionPath[1]} },
			vibrateActionState{ {&vibrateAction, &handSubactionPath[0]}, {&vibrateAction, &handSubactionPath[1]} },
			quitActionState{ &quitAction, nullptr },

			appSpace(&session, XR_REFERENCE_SPACE_TYPE_LOCAL),
			viewSpace(&session, XR_REFERENCE_SPACE_TYPE_VIEW),
			handSpace{ {&session, &poseAction, handSubactionPath[0]}, {&session, &poseAction, handSubactionPath[1]} },

			handScale{ 1.0f, 1.0f },
			handActive{ XR_FALSE, XR_FALSE },
			sessionRunning(false),
			exitRenderLoop(false),
			requestRestart(false),

			app(_app)
		{
			// print debug info
			instance.printInfo();
			session.printSystemInfo();
			session.printReferenceSpace();
			views.printViewConfig();
			views.printViewInfo();
			views.printSwapchainInfo();

			bind_controllers();
			actionSet.attach_session(&session);
		}

		void bind_controllers()
		{
			Controller<ControllerType::Simple> simple_controller(&instance);
			// Controller<ControllerType::GoogleDaydream> google_controller(&instance);
			Controller<ControllerType::HTCVive> vive_controller(&instance);
			// Controller<ControllerType::HTCVivePro> vive_pro_controller(&instance);
			Controller<ControllerType::MicrosoftMixedReality> msft_mr_controller(&instance);
			// Controller<ControllerType::MicrosoftXbox> msft_xbox_controller(&instance);
			// Controller<ControllerType::OculusGo> oculus_go_controller(&instance);
			Controller<ControllerType::OculusTouch> oculus_touch_controller(&instance);
			Controller<ControllerType::ValveIndex> valve_index_controller(&instance);

			bind_interaction_profile(simple_controller, {
				{grabAction, simple_controller.select_click[0]},
				{grabAction, simple_controller.select_click[1]},
				{poseAction, simple_controller.grip_pose[0]},
				{poseAction, simple_controller.grip_pose[1]},
				{quitAction, simple_controller.menu_click[0]},
				{quitAction, simple_controller.menu_click[1]},
				{vibrateAction, simple_controller.haptic[0]},
				{vibrateAction, simple_controller.haptic[1]} });

			bind_interaction_profile(vive_controller, {
				{grabAction, vive_controller.trigger_value[0]},
				{grabAction, vive_controller.trigger_value[1]},
				{poseAction, vive_controller.grip_pose[0]},
				{poseAction, vive_controller.grip_pose[1]},
				{quitAction, vive_controller.menu_click[0]},
				{quitAction, vive_controller.menu_click[1]},
				{vibrateAction, vive_controller.haptic[0]},
				{vibrateAction, vive_controller.haptic[1]} });

			bind_interaction_profile(msft_mr_controller, {
				{grabAction, msft_mr_controller.squeeze_click[0]},
				{grabAction, msft_mr_controller.squeeze_click[1]},
				{poseAction, msft_mr_controller.grip_pose[0]},
				{poseAction, msft_mr_controller.grip_pose[1]},
				{quitAction, msft_mr_controller.menu_click[0]},
				{quitAction, msft_mr_controller.menu_click[1]},
				{vibrateAction, msft_mr_controller.haptic[0]},
				{vibrateAction, msft_mr_controller.haptic[1]} });

			bind_interaction_profile(oculus_touch_controller, {
				{grabAction, oculus_touch_controller.squeeze_value[0]},
				{grabAction, oculus_touch_controller.squeeze_value[1]},
				{poseAction, oculus_touch_controller.grip_pose[0]},
				{poseAction, oculus_touch_controller.grip_pose[1]},
				{quitAction, oculus_touch_controller.menu_click},
				{vibrateAction, oculus_touch_controller.haptic[0]},
				{vibrateAction, oculus_touch_controller.haptic[1]} });

			bind_interaction_profile(valve_index_controller, {
				{grabAction, valve_index_controller.squeeze_force[0]},
				{grabAction, valve_index_controller.squeeze_force[1]},
				{poseAction, valve_index_controller.grip_pose[0]},
				{poseAction, valve_index_controller.grip_pose[1]},
				{quitAction, valve_index_controller.b_click[0]},
				{quitAction, valve_index_controller.b_click[1]},
				{vibrateAction, valve_index_controller.haptic[0]},
				{vibrateAction, valve_index_controller.haptic[1]} });
		}

		void handle_session_state_changed_event(XrEventDataSessionStateChanged const& stateChangedEvent)
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

		void log_action_source_name(XrAction action, const std::string& actionName)
		{
			XrBoundSourcesForActionEnumerateInfo getInfo = { XR_TYPE_BOUND_SOURCES_FOR_ACTION_ENUMERATE_INFO };
			getInfo.action = action;
			uint32_t pathCount = 0;
			xrEnumerateBoundSourcesForAction(session, &getInfo, 0, &pathCount, nullptr);
			std::vector<XrPath> paths(pathCount);
			xrEnumerateBoundSourcesForAction(session, &getInfo, uint32_t(paths.size()), &pathCount, paths.data());

			std::string sourceName;
			for (uint32_t i = 0; i < pathCount; ++i)
			{
				constexpr XrInputSourceLocalizedNameFlags all =
					XR_INPUT_SOURCE_LOCALIZED_NAME_USER_PATH_BIT |
					XR_INPUT_SOURCE_LOCALIZED_NAME_INTERACTION_PROFILE_BIT |
					XR_INPUT_SOURCE_LOCALIZED_NAME_COMPONENT_BIT;

				XrInputSourceLocalizedNameGetInfo nameInfo = { XR_TYPE_INPUT_SOURCE_LOCALIZED_NAME_GET_INFO };
				nameInfo.sourcePath = paths[i];
				nameInfo.whichComponents = all;

				uint32_t size = 0;
				xrGetInputSourceLocalizedName(session, &nameInfo, 0, &size, nullptr);
				if (size < 1)
					continue;
				std::vector<char> grabSource(size);
				xrGetInputSourceLocalizedName(session, &nameInfo, uint32_t(grabSource.size()), &size, grabSource.data());
				if (!sourceName.empty())
					sourceName += " and ";
				sourceName += "'";
				sourceName += std::string(grabSource.data(), size - 1);
				sourceName += "'";
			}

			printf("%s action is bound to %s\n", actionName.c_str(), ((!sourceName.empty()) ? sourceName.c_str() : "nothing"));
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
					handle_session_state_changed_event(sessionStateChangedEvent);
					break;
				}
				case XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED:
				{
					log_action_source_name(grabAction, "Grab");
					log_action_source_name(quitAction, "Quit");
					log_action_source_name(poseAction, "Pose");
					log_action_source_name(vibrateAction, "Vibrate");
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

		void poll_actions()
		{
			handActive[0] = XR_FALSE;
			handActive[1] = XR_FALSE;
			// Sync actions
			actionSet.sync();

			// Get pose and grab action state and start haptic vibrate when hand is 90% squeezed.
			for (auto hand : { 0, 1 })
			{
				auto& grab_state = grabActionState[hand];
				grab_state.update_state();

				auto& vibrate_state = vibrateActionState[hand];

				if (grab_state.is_active())
				{
					// Scale the rendered hand by 1.0f (open) to 0.5f (fully squeezed).
					handScale[hand] = 1.0f - 0.5f * grab_state.current_state();
					app->handScale = 0.03f * (1.f - grab_state.current_state());
					if (grab_state.current_state() > 0.9f)
					{
						vibrate_state.apply_haptic(0.5f, XR_MIN_HAPTIC_DURATION, XR_FREQUENCY_UNSPECIFIED);
					}
					else
					{
						//vibrate_state.stop_haptic();
					}
				}

				auto& pose_state = poseActionState[hand];
				pose_state.update_state();
				handActive[hand] = pose_state.is_active();
			}

			quitActionState.update_state();

			if (quitActionState.is_active() &&
				quitActionState.changed_since_last_sync() &&
				quitActionState.current_state())
			{
				xrRequestExitSession(session);
			}
		}

		bool render_layer(XrTime predictedDisplayTime,
			std::vector<XrCompositionLayerProjectionView>& projectionLayerViews,
			XrCompositionLayerProjection& layer)
		{
			if (!views.locate(predictedDisplayTime, appSpace))
				return false;

			projectionLayerViews.resize(views.validViewCount);

			std::vector<Cube> cubes;

			XrSpaceLocation viewSpaceLocation;
			XrResult res = viewSpace.locate_space(appSpace, predictedDisplayTime, &viewSpaceLocation);

			if (XR_UNQUALIFIED_SUCCESS(res))
			{
				if ((viewSpaceLocation.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0 &&
					(viewSpaceLocation.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) != 0)
				{
					cubes.push_back(Cube{ viewSpaceLocation.pose, {0.25f, 0.25f, 0.25f} });
				}
			}
			else
			{
				printf("Unable to locate a visualized reference space in app space: %d\n", res);
			}

			// Render a 10cm cube scaled by grabAction for each hand. Note renderHand will only be
			// true when the application has focus.
			for (auto hand : { 0, 1 })
			{
				XrSpaceLocation handLocation;
				res = handSpace[hand].locate_space(appSpace, predictedDisplayTime, &handLocation);
				if (XR_UNQUALIFIED_SUCCESS(res))
				{
					if ((handLocation.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0 &&
						(handLocation.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) != 0)
					{
						float scale = 0.1f * handScale[hand];
						app->hand2appPose = handLocation.pose;
						cubes.push_back(Cube{ handLocation.pose, {scale, scale, scale} });
					}
				}
				else
				{
					if (handActive[hand] == XR_TRUE)
					{
						const char* handName[] = { "left", "right" };
						printf("Unable to locate %s hand action space in app space: %d\n", handName[hand], res);
					}
				}
			}

			// Render view to the appropriate part of the swapchain image.
			for (uint32_t i = 0; i < views.validViewCount; i++)
			{
				// Each view has a separate swapchain which is acquired, rendered to, and released.
				//const Swapchain viewSwapchain = swapchains[i];

				auto& swapchain = views.swapchainImages[i];
				uint32_t swapchainImageIndex;

				swapchain.aquire(&swapchainImageIndex);
				swapchain.wait();

				projectionLayerViews[i] = { XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW };
				projectionLayerViews[i].pose = views.views[i].pose;
				projectionLayerViews[i].fov = views.views[i].fov;
				projectionLayerViews[i].subImage.swapchain = swapchain;
				projectionLayerViews[i].subImage.imageRect.offset = { 0, 0 };
				projectionLayerViews[i].subImage.imageRect.extent = { swapchain.width, swapchain.height };

				XrSwapchainImageOpenGLKHR const& swapchainImage = swapchain.swapchainImages[swapchainImageIndex];

				app->colorTexture = swapchainImage.image;
				app->viewport = projectionLayerViews[i].subImage.imageRect;
				app->fov = projectionLayerViews[i].fov;
				app->view2appPose = projectionLayerViews[i].pose;

				app->run();
				swapchain.release();
			}

			app->update_phy();

			layer.space = appSpace;
			layer.layerFlags = 0;
			layer.viewCount = (uint32_t)projectionLayerViews.size();
			layer.views = projectionLayerViews.data();
			return true;
		}

		void render()
		{
			frame.wait();
			frame.begin();

			std::vector<XrCompositionLayerBaseHeader*> layers;
			XrCompositionLayerProjection layer{ XR_TYPE_COMPOSITION_LAYER_PROJECTION };
			std::vector<XrCompositionLayerProjectionView> projectionLayerViews;

			if (frame.should_render())
			{
				if (render_layer(frame.predicted_display_time(), projectionLayerViews, layer))
				{
					layers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(&layer));
				}
			}

			frame.end(layers);
		}
	};
}

int main()
{
	GUI::UserInterface ui(Window::Window::Data{ "GalaxyInHand", { {800, 800}, /*resizable=*/true, /*fullscreen=*/false } }, false);
	OpenGL::HelloVR_GL renderer(30 * 1, false);
	ui.bindOpenGLMain(&renderer);
	ui.wm.swapInterval(0);
	ui.update();

	OpenXR::ApiLayer apiLayer;
	apiLayer.printInfo();

	OpenXR::GalaxyInHand galaxyInHand(ui.mainWindow->window.window, &renderer);

	for (;;)
	{
		galaxyInHand.poll_events();
		if (galaxyInHand.exitRenderLoop)
			break;
		galaxyInHand.poll_actions();
		galaxyInHand.render();
	}
}