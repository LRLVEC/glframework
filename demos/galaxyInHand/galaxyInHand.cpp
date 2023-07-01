#include <xr/_OpenXR.h>
#include <xr/_XrGui.h>
#include <_ImGui.h>
#include <_NBody.h>
#ifdef _CUDA
using NBodyImpl = OpenGL::NBodyCUDAImpl;
#else
using NBodyImpl = OpenGL::NBodyOpenGLImpl;
#endif
#include <array>
#include <map>

namespace OpenXR
{
	struct ActionStates
	{
		ActionState<ActionType::FloatInput> grabActionState[2];
		ActionState<ActionType::PoseInput> poseActionState[2];
		ActionState<ActionType::VibrationOutput> vibrateActionState[2];
		SpaceLocation handLocation[2];

		ActionStates(std::vector<Path>& handSubactionPath,
			Action* grabAction,
			Action* poseAction,
			Action* vibrateAction)
			:
			grabActionState{ {grabAction, &handSubactionPath[0]}, {grabAction, &handSubactionPath[1]} },
			poseActionState{ {poseAction, &handSubactionPath[0]}, {poseAction, &handSubactionPath[1]} },
			vibrateActionState{ {vibrateAction, &handSubactionPath[0]}, {vibrateAction, &handSubactionPath[1]} }
		{
		}
	};
}

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

	struct HelloVR : OpenGL, OpenXR::XrOpenGL
	{
		SourceManager sm;
		NBodyData nbodyData;
		NBodyImpl nbody;
		Math::mat4<float> view2proj;
		Math::mat4<float> app2view;
		Math::mat4<float> hand2app;
		Math::mat4<float> scale;
		Transform::BufferData trans;
		VRRenderer renderer;
		OpenXR::ActionStates* actionStates;
		float handScale;
		XrPosef hand2appPose;

		HelloVR(unsigned int _blocks, bool _experiment)
			:
			sm("./"),
			nbodyData(_blocks, _experiment, &sm),
			nbody(&nbodyData, &sm),
			renderer(&sm, &nbodyData.particlesArray, &trans),
			actionStates(nullptr),
			handScale(0.03f),
			hand2appPose{ 0 }
		{
			hand2appPose.orientation.w = 1.0f;
		}

		// OpenGL
		void init(FrameScale const& _size)override
		{
			glPointSize(1);
			glEnable(GL_DEPTH_TEST);
			renderer.transUniform.dataInit();
			renderer.particlesArray->dataInit();
			nbody.init();
		}
		void run()override
		{
			glClearColor(0.0f, 0.5f, 0.0f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT);
		}

		// xrOpenGL
		float nearZ(uint32_t eye)override
		{
			return 0.05f;
		}
		float farZ(uint32_t eye)override
		{
			return 20.f;
		}
		void pullActions()override
		{
			for (uint32_t hand(0); hand < 2; ++hand)
			{
				// hand scale
				if (actionStates->grabActionState[hand].is_active())
				{
					handScale = 0.03f * (1.f - actionStates->grabActionState[hand].current_state());
				}
				// hand location
				auto& hand_loc = actionStates->handLocation[hand];
				if (hand_loc.valid())
				{
					hand2appPose = hand_loc.location.pose;
				}
			}
			// hand scale
			scale.array[0][0] = handScale;
			scale.array[1][1] = handScale;
			scale.array[2][2] = handScale;
			scale.array[3][3] = 1.f;
			// hand location
			hand2app = OpenXR::get_transform(hand2appPose, false);
		}
		void update()override
		{
			nbody.run();
		}
		void setViewport(XrRect2Di const& viewport)override
		{
			glViewport(viewport.offset.x, viewport.offset.y, viewport.extent.width, viewport.extent.height);
		}
		void setViewMat(Math::mat4<float>const& _app2view)override
		{
			app2view = _app2view;
		}
		void setProjMat(Math::mat4<float>const& _view2proj)override
		{
			view2proj = _view2proj;
		}
		void xrRender(uint32_t eye)override
		{
			trans.ans = (view2proj, (app2view, (hand2app, scale)));
			renderer.use();
			renderer.run();
		}
	};
}

namespace OpenXR
{
	struct Galaxy :XrRunner
	{
		std::vector<Path> handSubactionPath;

		Action grabAction;
		Action poseAction;
		Action vibrateAction;
		Action quitAction;

		ActionStates actionStates;
		ActionState<ActionType::BoolInput> quitActionState;

		Space<SpaceType::Action> handSpace[2];
		XrBool32 handActive[2];

		Galaxy(GLFWwindow* _window)
			:
			XrRunner(_window),
			handSubactionPath{ {&instance, "/user/hand/left"}, {&instance, "/user/hand/right"} },

			grabAction(&actionSet, "grab_object", "Grab Object", XR_ACTION_TYPE_FLOAT_INPUT, handSubactionPath),
			poseAction(&actionSet, "hand_pose", "Hand Pose", XR_ACTION_TYPE_POSE_INPUT, handSubactionPath),
			vibrateAction(&actionSet, "vibrate_hand", "Vibrate Hand", XR_ACTION_TYPE_VIBRATION_OUTPUT, handSubactionPath),
			quitAction(&actionSet, "quit_session", "Quit Session", XR_ACTION_TYPE_BOOLEAN_INPUT),

			actionStates(handSubactionPath, &grabAction, &poseAction, &vibrateAction),
			quitActionState{ &quitAction, nullptr },

			handSpace{ {&session, &poseAction, handSubactionPath[0]}, {&session, &poseAction, handSubactionPath[1]} },

			handActive{ XR_FALSE, XR_FALSE }
		{
			bind_controllers();
			actionSet.attach_session(&session);
		}

		void bind_controllers()
		{
			bind_interaction_profile(controllerSet.simple_controller, {
				{grabAction, controllerSet.simple_controller.select_click[0]},
				{grabAction, controllerSet.simple_controller.select_click[1]},
				{poseAction, controllerSet.simple_controller.grip_pose[0]},
				{poseAction, controllerSet.simple_controller.grip_pose[1]},
				{quitAction, controllerSet.simple_controller.menu_click[0]},
				{quitAction, controllerSet.simple_controller.menu_click[1]},
				{vibrateAction, controllerSet.simple_controller.haptic[0]},
				{vibrateAction, controllerSet.simple_controller.haptic[1]} });

			bind_interaction_profile(controllerSet.vive_controller, {
				{grabAction, controllerSet.vive_controller.trigger_value[0]},
				{grabAction, controllerSet.vive_controller.trigger_value[1]},
				{poseAction, controllerSet.vive_controller.grip_pose[0]},
				{poseAction, controllerSet.vive_controller.grip_pose[1]},
				{quitAction, controllerSet.vive_controller.menu_click[0]},
				{quitAction, controllerSet.vive_controller.menu_click[1]},
				{vibrateAction, controllerSet.vive_controller.haptic[0]},
				{vibrateAction, controllerSet.vive_controller.haptic[1]} });

			bind_interaction_profile(controllerSet.msft_mr_controller, {
				{grabAction, controllerSet.msft_mr_controller.squeeze_click[0]},
				{grabAction, controllerSet.msft_mr_controller.squeeze_click[1]},
				{poseAction, controllerSet.msft_mr_controller.grip_pose[0]},
				{poseAction, controllerSet.msft_mr_controller.grip_pose[1]},
				{quitAction, controllerSet.msft_mr_controller.menu_click[0]},
				{quitAction, controllerSet.msft_mr_controller.menu_click[1]},
				{vibrateAction, controllerSet.msft_mr_controller.haptic[0]},
				{vibrateAction, controllerSet.msft_mr_controller.haptic[1]} });

			bind_interaction_profile(controllerSet.oculus_touch_controller, {
				{grabAction, controllerSet.oculus_touch_controller.squeeze_value[0]},
				{grabAction, controllerSet.oculus_touch_controller.squeeze_value[1]},
				{poseAction, controllerSet.oculus_touch_controller.grip_pose[0]},
				{poseAction, controllerSet.oculus_touch_controller.grip_pose[1]},
				{quitAction, controllerSet.oculus_touch_controller.menu_click},
				{vibrateAction, controllerSet.oculus_touch_controller.haptic[0]},
				{vibrateAction, controllerSet.oculus_touch_controller.haptic[1]} });

			bind_interaction_profile(controllerSet.valve_index_controller, {
				{grabAction, controllerSet.valve_index_controller.squeeze_force[0]},
				{grabAction, controllerSet.valve_index_controller.squeeze_force[1]},
				{poseAction, controllerSet.valve_index_controller.grip_pose[0]},
				{poseAction, controllerSet.valve_index_controller.grip_pose[1]},
				{quitAction, controllerSet.valve_index_controller.b_click[0]},
				{quitAction, controllerSet.valve_index_controller.b_click[1]},
				{vibrateAction, controllerSet.valve_index_controller.haptic[0]},
				{vibrateAction, controllerSet.valve_index_controller.haptic[1]} });
		}

		void update_actions()
		{
			actionSet.sync();

			// Get pose and grab action state and start haptic vibrate when hand is 90% squeezed.
			for (auto hand : { 0, 1 })
			{
				auto& grab_state = actionStates.grabActionState[hand];
				auto& vibrate_state = actionStates.vibrateActionState[hand];
				auto& pose_state = actionStates.poseActionState[hand];
				grab_state.update_state();
				pose_state.update_state();
				handActive[hand] = pose_state.is_active();
				if (grab_state.is_active())
				{
					// Scale the rendered hand by 1.0f (open) to 0.5f (fully squeezed).
					if (grab_state.current_state() > 0.9f)
						vibrate_state.apply_haptic(0.5f, XR_MIN_HAPTIC_DURATION, XR_FREQUENCY_UNSPECIFIED);
					else
					{
						//vibrate_state.stop_haptic();
					}
				}
			}

			quitActionState.update_state();
			if (quitActionState.is_active() &&
				quitActionState.changed_since_last_sync() &&
				quitActionState.current_state())
			{
				xrRequestExitSession(session);
			}
		}

		void before_render(XrTime predictedDisplayTime)
		{
			for (auto hand : { 0, 1 })
				handSpace[hand].locate_space(appSpace, predictedDisplayTime, &actionStates.handLocation[hand]);
		};
	};
}

int main()
{
	GUI::UserInterface ui(Window::Window::Data{ "GalaxyInHand", { {800, 800}, /*resizable=*/true, /*fullscreen=*/false } }, false);

	OpenGL::HelloVR helloVR(30 * 1, false);
	ui.bindOpenGLMain(&helloVR);
	ui.wm.swapInterval(0);
	ui.update();

	OpenXR::ApiLayer apiLayer;
	apiLayer.printInfo();
	OpenXR::Extension extension;
	extension.printInfo();

	OpenXR::Galaxy galaxy(ui.mainWindow->window.window);
	helloVR.actionStates = &galaxy.actionStates;
	galaxy.bindXrOpenGL(&helloVR);

	for (;;)
	{
		if (!ui.update())break;
		if (!galaxy.update())break;
	}
}