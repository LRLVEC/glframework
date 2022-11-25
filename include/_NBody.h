#pragma once
#include <_OpenGL.h>
#ifdef _VR
#include <GL/_OpenVR.h>
#endif
#include <time.h>
#include <random>
#include <unordered_map>

namespace CUDA
{
	struct Buffer;
}
struct NBodyCUDA_Glue;

namespace OpenGL
{
	struct NBodyData
	{
		struct Particles
		{
			struct Particle
			{
				Math::vec3<float>position;
				float mass;
				Math::vec4<float>velocity;
			};
			Vector<Particle>particles;
			std::mt19937 mt;
			std::uniform_real_distribution<float>randReal;
			unsigned int num;
			Particles() = delete;
			Particles(unsigned int _num)
				:
				mt(time(NULL)),
				num(_num),
				randReal(0, 1)
			{
			}
			Particle flatGalaxyParticles()
			{
				float r(100 * randReal(mt) + 0.1);
				float phi(2 * Math::Pi * randReal(mt));
				r = pow(r, 0.5);
				float vk(3.0f);
				float rn(0.3);
				return
				{
					{r * cos(phi),1.0f * randReal(mt) - 0.5f,r * sin(phi)},
					randReal(mt) > 0.999f ? 100 : randReal(mt),
					{-vk * sin(phi) / powf(r,rn),0,vk * cos(phi) / powf(r,rn)},
				};
			}
			Particle flatGalaxyParticlesOptimized(float blackHoleMass)
			{
				float r0(sqrtf(randReal(mt) + 0.01));
				float phi(2 * Math::Pi * randReal(mt));
				float r = r0 * 5;
				float vk(sqrtf(0.001f * (r * calcForce(r0) + blackHoleMass / r)));
				float m;
				if (randReal(mt) < 0.0001f)m = 1000 * randReal(mt);
				else m = randReal(mt);
				return
				{
					{r * cos(phi),0.2f * (randReal(mt) - 0.5f),r * sin(phi)},
					m,
					{-vk * sin(phi) ,0,vk * cos(phi) },
				};
			}
			Particle sphereGalaxyParticles()
			{
				float r(pow(100.0f * randReal(mt) + 0.1f, 1.0 / 3));
				float theta(2.0f * acos(randReal(mt)));
				float phi(2 * Math::Pi * randReal(mt));
				float vk(1.7f);
				float rn(0.5);
				return
				{
					{r * cos(phi) * sin(theta),r * sin(phi) * sin(theta),r * cos(theta)},
					randReal(mt) > 0.999f ? 100 : randReal(mt),
					{-vk * sin(phi) / powf(r,rn),vk * cos(phi) / powf(r,rn),0},
				};
			}
			Particle expFlatGalaxyParticles()
			{
				float r(100 * randReal(mt));
				float phi(2 * Math::Pi * randReal(mt));
				r = pow(r, 0.5);
				float vk(3.0f);
				float rn(0.3);
				return
				{
					{r * cos(phi),r * sin(phi),1.0f * randReal(mt) - 0.5f},
					randReal(mt),
					{-vk * sin(phi) / powf(r,rn),vk * cos(phi) / powf(r,rn),0},
				};
			}
			float calcForce(float r)
			{
				//r is in [0, 1], mass is uniformly distrubuted in [0, 1]
				return (0.00434f + r * (-0.03039f +
					r * (0.11616f + r * (-0.16195f + 0.08362f * r)))) * num;
			}
			void experimentGalaxy()
			{
				//This is used to create a distrubution without center black hole...
				//to see how force is distrubuted.
				unsigned int _num(num);
				while (_num--)particles.pushBack(expFlatGalaxyParticles());
			}
			void randomGalaxy()
			{
				unsigned int _num(num - 1);
				while (_num--)
					particles.pushBack(flatGalaxyParticles());
				particles.pushBack
				(
					{
						{0,0,0},
						8000,
						{0,0,0},
					}
				);
			}
			void randomGalaxyOptimized()
			{
				unsigned int _num(num - 1);
				float blackHoleMass(200000.0f);
				while (_num--)
					particles.pushBack(flatGalaxyParticlesOptimized(blackHoleMass));
				particles.pushBack
				(
					{
						{0,0,0},
						blackHoleMass,
						{0,0,0},
					}
				);
			}
		};
		struct ParticlesData : Buffer::Data
		{
			Particles* particles;
			ParticlesData(Particles* _particles)
				:
				Data(DynamicDraw),
				particles(_particles)
			{
			}
			virtual void* pointer()override
			{
				return particles->particles.data;
			}
			virtual unsigned int size()override
			{
				return sizeof(Particles::Particle) * (particles->particles.length);
			}
		};

		SourceManager* sm;
		Particles particles;
		ParticlesData particlesData;
		Buffer particlesBuffer;
		BufferConfig particlesArray;
		bool experiment;

		NBodyData(unsigned int _blocks, bool _experiment, SourceManager* _sm)
			:
			sm(_sm),
			particles(_blocks * 1024),
			particlesData(&particles),
			particlesBuffer(&particlesData),
			particlesArray(&particlesBuffer, ArrayBuffer),
			experiment(_experiment)
		{
			if (_experiment)
				particles.experimentGalaxy();
			else
				particles.randomGalaxyOptimized();
		}
	};

	struct NBodyImplBase
	{
		NBodyData* nbodyData;
		NBodyImplBase(NBodyData* _nbodyData) : nbodyData(_nbodyData) {}
		virtual void init() = 0;
		virtual void run() = 0;
		virtual void close() = 0;
	};

	struct NBodySingleViewRenderer : OpenGL
	{
		struct Renderer : Program
		{
			BufferConfig* particlesArray;
			Buffer transBuffer;
			BufferConfig transUniform;
			VertexAttrib positions;
			VertexAttrib velocities;

			Renderer(SourceManager* _sm, BufferConfig* _particlesArray, Transform* _trans)
				:
				Program(_sm, "Renderer", Vector< VertexAttrib*>{&positions, & velocities}),
				particlesArray(_particlesArray),
				transBuffer(&_trans->bufferData),
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
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				glDrawArrays(GL_POINTS, 0, particlesArray->buffer->data->size() / sizeof(NBodyData::Particles::Particle));
			}
		};

		NBodyImplBase* nbody;

		Transform trans;
		Renderer renderer;

		NBodySingleViewRenderer(NBodyImplBase* _nbody, SourceManager* _sm)
			:
			nbody(_nbody),
			trans({ {80.0,0.1,800},{0.8,0.8,0.1},{1},500.0 }),
			renderer(_sm, &nbody->nbodyData->particlesArray, &trans)
		{
		}
		virtual void init(FrameScale const& _size) override
		{
			glViewport(0, 0, _size.w, _size.h);
			glPointSize(2);
			glEnable(GL_DEPTH_TEST);
			trans.init(_size);
			renderer.transUniform.dataInit();
			renderer.particlesArray->dataInit();
			nbody->init();
		}
		virtual void run() override
		{
			glfwGetCurrentContext();
			trans.operate();
			if (trans.updated)
			{
				renderer.transUniform.refreshData();
				trans.updated = false;
			}
			renderer.use();
			renderer.run();
			nbody->run();
		}
		virtual void frameSize(GLFWwindow* _window, int _w, int _h) override
		{
			trans.resize(_w, _h);
			glViewport(0, 0, _w, _h);
		}
		virtual void mouseButton(GLFWwindow* _window, int _button, int _action, int _mods) override
		{
			switch (_button)
			{
			case GLFW_MOUSE_BUTTON_LEFT:trans.mouse.refreshButton(0, _action); break;
			case GLFW_MOUSE_BUTTON_MIDDLE:trans.mouse.refreshButton(1, _action); break;
			case GLFW_MOUSE_BUTTON_RIGHT:trans.mouse.refreshButton(2, _action); break;
			}
		}
		virtual void mousePos(GLFWwindow* _window, double _x, double _y)
		{
			trans.mouse.refreshPos(_x, _y);
		}
		virtual void mouseScroll(GLFWwindow* _window, double _x, double _y)
		{
			if (_y != 0.0)
				trans.scroll.refresh(_y);
		}
		virtual void key(GLFWwindow* _window, int _key, int _scancode, int _action, int _mods)
		{
			switch (_key)
			{
			case GLFW_KEY_ESCAPE:
				if (_action == GLFW_PRESS)
					glfwSetWindowShouldClose(_window, true);
				nbody->close();
				break;
			case GLFW_KEY_A:trans.key.refresh(0, _action); break;
			case GLFW_KEY_D:trans.key.refresh(1, _action); break;
			case GLFW_KEY_W:trans.key.refresh(2, _action); break;
			case GLFW_KEY_S:trans.key.refresh(3, _action); break;
			}
		}
	};

	struct NBodyMultiViewRenderer : OpenGL
	{
		struct Renderer : Program
		{
			BufferConfig* particlesArray;
			Buffer transBuffer;
			BufferConfig transUniform;
			VertexAttrib positions;
			VertexAttrib velocities;

			Renderer(SourceManager* _sm, BufferConfig* _particlesArray)
				:
				Program(_sm, "Renderer", Vector< VertexAttrib*>{&positions, & velocities}),
				particlesArray(_particlesArray),
				transBuffer(nullptr),
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
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				glDrawArrays(GL_POINTS, 0, particlesArray->buffer->data->size() / sizeof(NBodyData::Particles::Particle));
			}
		};

		NBodyImplBase* nbody;
		// Transform trans;
		GLFWwindow* mainWindow;
		std::unordered_map<GLFWwindow*, Transform*> transforms;
		Renderer renderer;

		NBodyMultiViewRenderer(NBodyImplBase* _nbody, SourceManager* _sm)
			:
			nbody(_nbody),
			renderer(_sm, &nbody->nbodyData->particlesArray)
		{
		}
		~NBodyMultiViewRenderer()
		{
			for (auto d : transforms)
			{
				delete d.second;
			}
		}
		void registerMainTransform(GLFWwindow* _window)
		{
			mainWindow = _window;
			FrameScale size;
			glfwGetWindowSize(_window, &size.w, &size.h);
			transforms[_window] = new Transform({ {80.0,0.1,800},{0.8,0.8,0.1},{1},500.0 });
			// transforms[_window]->init(size);
		}
		void registerTransform(GLFWwindow* _window)
		{
			FrameScale size;
			glfwGetWindowSize(_window, &size.w, &size.h);
			transforms[_window] = new Transform({ {80.0,0.1,800},{0.8,0.8,0.1},{1},500.0 });
			transforms[_window]->init(size);
		}
		void unregisterTransform(GLFWwindow* _window)
		{
			delete transforms[_window];
			transforms.erase(_window);
			if (_window == mainWindow)
			{
				mainWindow = nullptr;
			}
		}
		bool isTransformRegistered(GLFWwindow* _window)
		{
			return transforms.count(_window);
		}
		Transform* getTrans(GLFWwindow* _window = nullptr)
		{
			if (!_window)
			{
				_window = glfwGetCurrentContext();
			}
			if (transforms.count(_window))
			{
				return transforms[_window];
			}
			else
			{
				printf("Trys to get trans from unregistered window %p!\n", _window);
				throw std::runtime_error{ "Unregistered window!" };
				return nullptr;
			}
		}
		void switchToMainOpenGLContext()
		{
			// detaches original ogl context
			glfwMakeContextCurrent(mainWindow);
		}
		void setRenderTarget(GLFWwindow* _window)
		{
			wglMakeCurrent(GetDC(glfwGetWin32Window(_window)), glfwGetWGLContext(mainWindow));
		}
		virtual void init(FrameScale const& _size) override
		{
			GLFWwindow* target = glfwGetCurrentContext();
			//printf("Init from %p\n", target);
			switchToMainOpenGLContext();
			setRenderTarget(target);
			// printf("Init from %p\n", glfwGetCurrentContext());
			glViewport(0, 0, _size.w, _size.h);
			glPointSize(2);
			glEnable(GL_DEPTH_TEST);
			Transform* trans(getTrans(target));
			trans->init(_size);
			renderer.transBuffer.data = &trans->bufferData;
			renderer.transUniform.dataInit();
			glfwMakeContextCurrent(target);
			if (glfwGetCurrentContext() == mainWindow)
			{
				renderer.particlesArray->dataInit();
				nbody->init();
			}
		}
		virtual void run() override
		{
			GLFWwindow* target = glfwGetCurrentContext();
			//printf("Run from %p\n", target);
			FrameScale frameSize;
			glfwGetWindowSize(target, &frameSize.w, &frameSize.h);
			if (frameSize.w && frameSize.h)
			{
				switchToMainOpenGLContext();
				setRenderTarget(target);
				Transform* trans(getTrans(target));
				trans->operate();
				renderer.transBuffer.data = &trans->bufferData;
				renderer.transUniform.refreshData();
				trans->updated = false;
				glViewport(0, 0, frameSize.w, frameSize.h);
				renderer.use();
				renderer.run();
				glfwMakeContextCurrent(target);
			}
			if (glfwGetCurrentContext() == mainWindow)
			{
				nbody->run();
			}
		}
		virtual void frameSize(GLFWwindow* _window, int _w, int _h) override
		{
			//printf("FrameSize from %p %d %d\n", _window, _w, _h);
			if (_w && _h)
			{
				getTrans(_window)->resize(_w, _h);
			}
		}
		virtual void mouseButton(GLFWwindow* _window, int _button, int _action, int _mods) override
		{
			if (glfwWindowShouldClose(_window) || !isTransformRegistered(_window))
			{
				return;
			}
			Transform* trans(getTrans(_window));
			switch (_button)
			{
			case GLFW_MOUSE_BUTTON_LEFT:trans->mouse.refreshButton(0, _action); break;
			case GLFW_MOUSE_BUTTON_MIDDLE:trans->mouse.refreshButton(1, _action); break;
			case GLFW_MOUSE_BUTTON_RIGHT:trans->mouse.refreshButton(2, _action); break;
			}
		}
		virtual void mousePos(GLFWwindow* _window, double _x, double _y)
		{
			if (glfwWindowShouldClose(_window) || !isTransformRegistered(_window))
			{
				return;
			}
			Transform* trans(getTrans(_window));
			trans->mouse.refreshPos(_x, _y);
		}
		virtual void mouseScroll(GLFWwindow* _window, double _x, double _y)
		{
			if (glfwWindowShouldClose(_window) || !isTransformRegistered(_window))
			{
				return;
			}
			Transform* trans(getTrans(_window));
			if (_y != 0.0)
				trans->scroll.refresh(_y);
		}
		virtual void key(GLFWwindow* _window, int _key, int _scancode, int _action, int _mods)
		{
			//printf("Key from %p\n", _window);
			//printf("Registered: %d\n", isTransformRegistered(_window));
			if (glfwWindowShouldClose(_window) || !isTransformRegistered(_window))
			{
				return;
			}
			Transform* trans(getTrans(_window));
			switch (_key)
			{
			case GLFW_KEY_ESCAPE:
				if (_action == GLFW_PRESS)
				{
					glfwSetWindowShouldClose(_window, true);
					if (_window == mainWindow)
					{
						nbody->close();
					}
					unregisterTransform(_window);
				}
				break;
			case GLFW_KEY_A:trans->key.refresh(0, _action); break;
			case GLFW_KEY_D:trans->key.refresh(1, _action); break;
			case GLFW_KEY_W:trans->key.refresh(2, _action); break;
			case GLFW_KEY_S:trans->key.refresh(3, _action); break;
			}
		}
	};

	struct NBodyOpenGLImpl : NBodyImplBase
	{
		struct ComputeParticles :Computers
		{
			struct ParameterData : Buffer::Data
			{
				struct Parameter
				{
					float dt;
					float G;
					unsigned int num;
				};
				Parameter parameter;
				ParameterData(Parameter const& _parameter)
					:
					parameter(_parameter)
				{

				}
				virtual unsigned int size()override
				{
					return sizeof(Parameter);
				}
				virtual void* pointer()override
				{
					return &parameter;
				}
			};
			struct VelocityCalculation :Program
			{
				ParameterData* parameterData;
				VelocityCalculation(SourceManager* _sm, ParameterData* _parameterData)
					:
					Program(_sm, "VelocityCalculation"),
					parameterData(_parameterData)
				{
					init();
				}
				virtual void initBufferData()override
				{
				}
				virtual void run()override
				{
					glDispatchCompute(parameterData->parameter.num / 1024, 1, 1);
				}
			};
			struct PositionCalculation :Program
			{
				ParameterData* parameterData;
				PositionCalculation(SourceManager* _sm, ParameterData* _parameterData)
					:
					Program(_sm, "PositionCalculation"),
					parameterData(_parameterData)
				{
					init();
				}
				virtual void initBufferData()override
				{
				}
				virtual void run()override
				{
					glDispatchCompute(parameterData->parameter.num / 1024, 1, 1);
				}
			};

			BufferConfig particlesStorage;
			ParameterData parameterData;
			Buffer parameterBuffer;
			BufferConfig parameterUniform;
			VelocityCalculation velocityCalculation;
			PositionCalculation positionCalculation;
			ComputeParticles(SourceManager* _sm, Buffer* _particlesBuffer, NBodyData::Particles* _particles)
				:
				particlesStorage(_particlesBuffer, ShaderStorageBuffer, 1),
				parameterData({ 0.005f,0.001f,_particles->num }),
				parameterBuffer(&parameterData),
				parameterUniform(&parameterBuffer, UniformBuffer, 3),
				velocityCalculation(_sm, &parameterData),
				positionCalculation(_sm, &parameterData)
			{
			}
			virtual void initBufferData()override
			{
			}
			virtual void run()override
			{
				//particlesStorage.bind();
				velocityCalculation.use();
				velocityCalculation.run();
				glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
				positionCalculation.use();
				positionCalculation.run();
				glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
			}
			void init()
			{
				parameterUniform.dataInit();
			}
		};

		ComputeParticles computeParticles;
		NBodyOpenGLImpl(NBodyData* _nbodyData, SourceManager* _sm)
			:
			NBodyImplBase(_nbodyData),
			computeParticles(_sm, &_nbodyData->particlesBuffer, &_nbodyData->particles)
		{
		}
		void init() override
		{
			computeParticles.init();
		}
		void run() override
		{
			computeParticles.run();
		}
		void close() override
		{
		}
	};

#ifdef _VR
	namespace VR
	{
		struct NBodyVR :OpenGL
		{
			struct Particles
			{
				struct Particle
				{
					Math::vec3<float>position;
					float mass;
					Math::vec4<float>velocity;
				};
				Vector<Particle>particles;
				std::mt19937 mt;
				std::uniform_real_distribution<float>randReal;
				unsigned int num;
				Particles() = delete;
				Particles(unsigned int _num)
					:
					mt(time(NULL)),
					num(_num),
					randReal(0, 1)
				{
				}
				Particle flatGalaxyParticles()
				{
					float r(100 * randReal(mt) + 0.1);
					float phi(2 * Math::Pi * randReal(mt));
					r = pow(r, 0.5);
					float vk(3.0f);
					float rn(0.3);
					return
					{
						{r * cos(phi),1.0f * randReal(mt) - 0.5f,r * sin(phi)},
						randReal(mt) > 0.999f ? 100 : randReal(mt),
						{-vk * sin(phi) / powf(r,rn),0,vk * cos(phi) / powf(r,rn)},
					};
				}
				Particle flatGalaxyParticlesOptimized(float blackHoleMass)
				{
					float r0(sqrtf(randReal(mt) + 0.01));
					float phi(2 * Math::Pi * randReal(mt));
					float r = r0 * 0.1;
					float vk(sqrtf(0.001f * (r * calcForce(r0) + blackHoleMass / r)));
					return
					{
						{r * cos(phi),.002f * randReal(mt) + 0.899f,r * sin(phi)},
						randReal(mt),
						{-vk * sin(phi) ,0,vk * cos(phi) },
					};
				}
				Particle sphereGalaxyParticles()
				{
					float r(pow(100.0f * randReal(mt) + 0.1f, 1.0 / 3));
					float theta(2.0f * acos(randReal(mt)));
					float phi(2 * Math::Pi * randReal(mt));
					float vk(1.7f);
					float rn(0.5);
					return
					{
						{r * cos(phi) * sin(theta),r * sin(phi) * sin(theta),r * cos(theta)},
						randReal(mt) > 0.999f ? 100 : randReal(mt),
						{-vk * sin(phi) / powf(r,rn),vk * cos(phi) / powf(r,rn),0},
					};
				}
				Particle expFlatGalaxyParticles()
				{
					float r(100 * randReal(mt));
					float phi(2 * Math::Pi * randReal(mt));
					r = pow(r, 0.5);
					float vk(3.0f);
					float rn(0.3);
					return
					{
						{r * cos(phi),r * sin(phi),1.0f * randReal(mt) - 0.5f},
						randReal(mt),
						{-vk * sin(phi) / powf(r,rn),vk * cos(phi) / powf(r,rn),0},
					};
				}
				float calcForce(float r)
				{
					//r is in [0, 1], mass is uniformly distrubuted in [0, 1]
					return (0.00434f + r * (-0.03039f +
						r * (0.11616f + r * (-0.16195f + 0.08362f * r)))) * num;
				}
				void experimentGalaxy()
				{
					//This is used to create a distrubution without center black hole...
					//to see how force is distrubuted.
					unsigned int _num(num);
					while (_num--)particles.pushBack(expFlatGalaxyParticles());
				}
				void randomGalaxy()
				{
					unsigned int _num(num - 1);
					while (_num--)
						particles.pushBack(flatGalaxyParticles());
					particles.pushBack
					(
						{
							{0,0,0},
							8000,
							{0,0,0},
						}
					);
				}
				void randomGalaxyOptimized()
				{
					unsigned int _num(num - 1);
					float blackHoleMass(200000.0f);
					while (_num--)
						particles.pushBack(flatGalaxyParticlesOptimized(blackHoleMass));
					particles.pushBack
					(
						{
							{0,0.9,0},
							blackHoleMass,
							{0,0,0},
						}
					);
				}
			};
			struct ParticlesData :Buffer::Data
			{
				Particles* particles;
				ParticlesData(Particles* _particles)
					:
					Data(DynamicDraw),
					particles(_particles)
				{
				}
				virtual void* pointer()override
				{
					return particles->particles.data;
				}
				virtual unsigned int size()override
				{
					return sizeof(Particles::Particle) * (particles->particles.length);
				}
			};

			struct Renderer :Program
			{
				Transform* trans;
				Buffer transBuffer;
				BufferConfig transUniform;
				BufferConfig particlesArray;
				VertexAttrib positions;
				VertexAttrib velocities;
				VRDevice* hmd;
				Trans* vrTrans;
				FrameBufferDesc leftEyeDesc;
				FrameBufferDesc rightEyeDesc;
				FrameScale windowSize;

				Renderer(SourceManager* _sm, Buffer* _particlesBuffer, Transform* _trans, Trans* _vrTrans, VRDevice* _hmd)
					:
					Program(_sm, "Renderer", Vector<VertexAttrib*>{&positions, & velocities}),
					trans(_trans),
					transBuffer(&_trans->bufferData),
					transUniform(&transBuffer, UniformBuffer, 0),
					particlesArray(_particlesBuffer, ArrayBuffer),
					positions(&particlesArray, 0, VertexAttrib::three, VertexAttrib::Float, false, sizeof(Particles::Particle), 0, 0),
					velocities(&particlesArray, 1, VertexAttrib::three, VertexAttrib::Float, false, sizeof(Particles::Particle), 16, 0),
					hmd(_hmd),
					vrTrans(_vrTrans),
					leftEyeDesc(hmd->frameScale),
					rightEyeDesc(hmd->frameScale)
				{
					init();
				}
				virtual void initBufferData()override
				{
				}
				virtual void run()override
				{
					glClearColor(0.f, 0.f, 0.f, 1.0f);
					glEnable(GL_MULTISAMPLE);

					// Left Eye
					glBindFramebuffer(GL_FRAMEBUFFER, leftEyeDesc.renderFramebuffer);
					glViewport(0, 0, hmd->frameScale.w, hmd->frameScale.h);
					glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
					transBuffer.data = &vrTrans->leftEye;
					transUniform.refreshData();
					glDrawArrays(GL_POINTS, 0, particlesArray.buffer->data->size() / sizeof(Particles::Particle));
					// Right Eye
					glBindFramebuffer(GL_FRAMEBUFFER, rightEyeDesc.renderFramebuffer);
					glViewport(0, 0, hmd->frameScale.w, hmd->frameScale.h);
					glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
					transBuffer.data = &vrTrans->rightEye;
					transUniform.refreshData();
					glDrawArrays(GL_POINTS, 0, particlesArray.buffer->data->size() / sizeof(Particles::Particle));
					glBindFramebuffer(GL_FRAMEBUFFER, 0);

					glDisable(GL_MULTISAMPLE);
					leftEyeDesc.copyRenderBuffer();
					rightEyeDesc.copyRenderBuffer();

					vr::Texture_t leftEyeTexture = { (void*)(uintptr_t)leftEyeDesc.resolveTexture,
						vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
					vr::VRCompositor()->Submit(vr::Eye_Left, &leftEyeTexture);
					vr::Texture_t rightEyeTexture = { (void*)(uintptr_t)rightEyeDesc.resolveTexture,
						vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
					vr::VRCompositor()->Submit(vr::Eye_Right, &rightEyeTexture);
					glFlush();

					glViewport(0, 0, windowSize.w, windowSize.h);
					glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
					glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
					transBuffer.data = &trans->bufferData;
					transUniform.refreshData();
					glDrawArrays(GL_POINTS, 0, particlesArray.buffer->data->size() / sizeof(Particles::Particle));
				}
			};
			struct ComputeParticles :Computers
			{
				struct ParameterData : Buffer::Data
				{
					struct Parameter
					{
						float dt;
						float G;
						unsigned int num;
					};
					Parameter parameter;
					ParameterData(Parameter const& _parameter)
						:
						parameter(_parameter)
					{

					}
					virtual unsigned int size()override
					{
						return sizeof(Parameter);
					}
					virtual void* pointer()override
					{
						return &parameter;
					}
				};
				struct VelocityCalculation :Program
				{
					ParameterData* parameterData;
					VelocityCalculation(SourceManager* _sm, ParameterData* _parameterData)
						:
						Program(_sm, "VelocityCalculation"),
						parameterData(_parameterData)
					{
						init();
					}
					virtual void initBufferData()override
					{
					}
					virtual void run()override
					{
						glDispatchCompute(parameterData->parameter.num / 1024, 1, 1);
					}
				};
				struct PositionCalculation :Program
				{
					ParameterData* parameterData;
					PositionCalculation(SourceManager* _sm, ParameterData* _parameterData)
						:
						Program(_sm, "PositionCalculation"),
						parameterData(_parameterData)
					{
						init();
					}
					virtual void initBufferData()override
					{
					}
					virtual void run()override
					{
						glDispatchCompute(parameterData->parameter.num / 1024, 1, 1);
					}
				};

				BufferConfig particlesStorage;
				ParameterData parameterData;
				Buffer parameterBuffer;
				BufferConfig parameterUniform;
				VelocityCalculation velocityCalculation;
				PositionCalculation positionCalculation;
				ComputeParticles(SourceManager* _sm, Buffer* _particlesBuffer, Particles* _particles)
					:
					particlesStorage(_particlesBuffer, ShaderStorageBuffer, 1),
					parameterData({ 0.00001f,0.001f,_particles->num }),
					parameterBuffer(&parameterData),
					parameterUniform(&parameterBuffer, UniformBuffer, 3),
					velocityCalculation(_sm, &parameterData),
					positionCalculation(_sm, &parameterData)
				{
				}
				virtual void initBufferData()override
				{
				}
				virtual void run()override
				{
					//particlesStorage.bind();
					velocityCalculation.use();
					velocityCalculation.run();
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
					positionCalculation.use();
					positionCalculation.run();
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
				}
				void init()
				{
					parameterUniform.dataInit();
				}
			};

			SourceManager sm;
			Particles particles;
			ParticlesData particlesData;
			Buffer particlesBuffer;
			VRDevice hmd;
			Trans vrTrans;
			Transform trans;
			Renderer renderer;
			ComputeParticles computeParticles;

			NBodyVR(unsigned int _groups)
				:
				sm(),
				particles(_groups << 10),
				particlesData(&particles),
				particlesBuffer(&particlesData),
				hmd(false),
				vrTrans(&hmd, { 0.01, 30 }),
				trans({ {80.0,0.1,800},{0.01,0.9,0.001},{0.01},500.0 }),
				renderer(&sm, &particlesBuffer, &trans, &vrTrans, &hmd),
				computeParticles(&sm, &particlesBuffer, &particles)
			{
				particles.randomGalaxyOptimized();
			}
			virtual void init(FrameScale const& _size)override
			{
				glPointSize(2);
				glEnable(GL_DEPTH_TEST);
				trans.init(_size);
				renderer.windowSize = _size;
				renderer.transUniform.dataInit();
				renderer.particlesArray.dataInit();
				computeParticles.init();
			}
			virtual void run()override
			{
				vrTrans.update();
				trans.operate();
				if (trans.updated)
				{
					trans.updated = false;
				}
				renderer.use();
				renderer.run();
				computeParticles.run();
			}
			virtual void frameSize(int _w, int _h) override
			{
				trans.resize(_w, _h);
				renderer.windowSize = { _w,_h };
			}
			virtual void framePos(int, int) override
			{
			}
			virtual void frameFocus(int) override
			{
			}
			virtual void mouseButton(int _button, int _action, int _mods) override
			{
				switch (_button)
				{
				case GLFW_MOUSE_BUTTON_LEFT:trans.mouse.refreshButton(0, _action); break;
				case GLFW_MOUSE_BUTTON_MIDDLE:trans.mouse.refreshButton(1, _action); break;
				case GLFW_MOUSE_BUTTON_RIGHT:trans.mouse.refreshButton(2, _action); break;
				}
			}
			virtual void mousePos(double _x, double _y) override
			{
				trans.mouse.refreshPos(_x, _y);
			}
			virtual void mouseScroll(double _x, double _y)override
			{
				if (_y != 0.0)
					trans.scroll.refresh(_y);
			}
			virtual void key(GLFWwindow* _window, int _key, int _scancode, int _action, int _mods) override
			{
				switch (_key)
				{
				case GLFW_KEY_ESCAPE:
					if (_action == GLFW_PRESS)
						glfwSetWindowShouldClose(_window, true);
					break;
				case GLFW_KEY_A:trans.key.refresh(0, _action); break;
				case GLFW_KEY_D:trans.key.refresh(1, _action); break;
				case GLFW_KEY_W:trans.key.refresh(2, _action); break;
				case GLFW_KEY_S:trans.key.refresh(3, _action); break;
		}
	}
};
	}
#endif
#ifdef _CUDA
	// OpenGL compute shader implement
	struct NBodyCUDAImpl : NBodyImplBase
	{
		CUDA::Buffer* particlesBufferCUDA;
		NBodyCUDA_Glue* glue;

		NBodyCUDAImpl(NBodyData* _nbodyData, SourceManager* _sm);
		~NBodyCUDAImpl();
		void experiment();
		virtual void init()override;
		virtual void run()override;
		virtual void close() override;
	};
#ifdef _VR
	namespace VR
	{
		struct NBodyCUDAVR : OpenGL
		{
			struct Particles
			{
				struct Particle
				{
					Math::vec3<float>position;
					float mass;
					Math::vec4<float>velocity;
				};
				Vector<Particle>particles;
				std::mt19937 mt;
				std::uniform_real_distribution<float>randReal;
				unsigned int num;
				Particles() = delete;
				Particles(unsigned int _num)
					:
					mt(time(NULL)),
					num(_num),
					randReal(0, 1)
				{
				}
				Particle flatGalaxyParticles()
				{
					float r(100 * randReal(mt) + 0.1);
					float phi(2 * Math::Pi * randReal(mt));
					r = pow(r, 0.5);
					float vk(3.0f);
					float rn(0.3);
					return
					{
						{r * cos(phi),1.0f * randReal(mt) - 0.5f,r * sin(phi)},
						randReal(mt) > 0.999f ? 100 : randReal(mt),
						{-vk * sin(phi) / powf(r,rn),0,vk * cos(phi) / powf(r,rn)},
					};
				}
				Particle flatGalaxyParticlesOptimized(float blackHoleMass)
				{
					float r0(sqrtf(randReal(mt) + 0.01));
					float phi(2 * Math::Pi * randReal(mt));
					float r = r0 * 0.1;
					float vk(sqrtf(0.001f * (r * calcForce(r0) + blackHoleMass / r)));
					return
					{
						{r * cos(phi),.002f * randReal(mt) + 0.899f,r * sin(phi)},
						randReal(mt),
						{-vk * sin(phi) ,0,vk * cos(phi) },
					};
				}
				Particle sphereGalaxyParticles()
				{
					float r(pow(100.0f * randReal(mt) + 0.1f, 1.0 / 3));
					float theta(2.0f * acos(randReal(mt)));
					float phi(2 * Math::Pi * randReal(mt));
					float vk(1.7f);
					float rn(0.5);
					return
					{
						{r * cos(phi) * sin(theta),r * sin(phi) * sin(theta),r * cos(theta)},
						randReal(mt) > 0.999f ? 100 : randReal(mt),
						{-vk * sin(phi) / powf(r,rn),vk * cos(phi) / powf(r,rn),0},
					};
				}
				Particle expFlatGalaxyParticles()
				{
					float r(100 * randReal(mt));
					float phi(2 * Math::Pi * randReal(mt));
					r = pow(r, 0.5);
					float vk(3.0f);
					float rn(0.3);
					return
					{
						{r * cos(phi),r * sin(phi),1.0f * randReal(mt) - 0.5f},
						randReal(mt),
						{-vk * sin(phi) / powf(r,rn),vk * cos(phi) / powf(r,rn),0},
					};
				}
				float calcForce(float r)
				{
					//r is in [0, 1], mass is uniformly distrubuted in [0, 1]
					return (0.00434f + r * (-0.03039f +
						r * (0.11616f + r * (-0.16195f + 0.08362f * r)))) * num;
				}
				void experimentGalaxy()
				{
					//This is used to create a distrubution without center black hole...
					//to see how force is distrubuted.
					unsigned int _num(num);
					while (_num--)particles.pushBack(expFlatGalaxyParticles());
				}
				void randomGalaxy()
				{
					unsigned int _num(num - 1);
					while (_num--)
						particles.pushBack(flatGalaxyParticles());
					particles.pushBack
					(
						{
							{0,0,0},
							8000,
							{0,0,0},
						}
					);
				}
				void randomGalaxyOptimized()
				{
					unsigned int _num(num - 1);
					float blackHoleMass(200000.0f);
					while (_num--)
						particles.pushBack(flatGalaxyParticlesOptimized(blackHoleMass));
					particles.pushBack
					(
						{
							{0,0.9,0},
							blackHoleMass,
							{0,0,0},
						}
					);
				}
			};
			struct ParticlesData : Buffer::Data
			{
				Particles* particles;
				ParticlesData(Particles* _particles)
					:
					Data(DynamicDraw),
					particles(_particles)
				{
				}
				virtual void* pointer()override
				{
					return particles->particles.data;
				}
				virtual unsigned int size()override
				{
					return sizeof(Particles::Particle) * (particles->particles.length);
				}
			};
			struct Renderer : Program
			{
				Transform* trans;
				Buffer transBuffer;
				BufferConfig transUniform;
				BufferConfig particlesArray;
				VertexAttrib positions;
				VertexAttrib velocities;
				VRDevice* hmd;
				Trans* vrTrans;
				FrameBufferDesc leftEyeDesc;
				FrameBufferDesc rightEyeDesc;
				FrameScale windowSize;

				Renderer(SourceManager* _sm, Buffer* _particlesBuffer, Transform* _trans, Trans* _vrTrans, VRDevice* _hmd)
					:
					Program(_sm, "Renderer", Vector< VertexAttrib*>{&positions, & velocities}),
					trans(_trans),
					transBuffer(&_trans->bufferData),
					transUniform(&transBuffer, UniformBuffer, 0),
					particlesArray(_particlesBuffer, ArrayBuffer),
					positions(&particlesArray, 0, VertexAttrib::three, VertexAttrib::Float, false, sizeof(Particles::Particle), 0, 0),
					velocities(&particlesArray, 1, VertexAttrib::three, VertexAttrib::Float, false, sizeof(Particles::Particle), 16, 0),
					hmd(_hmd),
					vrTrans(_vrTrans),
					leftEyeDesc(hmd->frameScale),
					rightEyeDesc(hmd->frameScale)
				{
					init();
				}
				virtual void initBufferData()override
				{
				}
				virtual void run()override
				{
					glClearColor(0.f, 0.f, 0.f, 1.0f);

					// Left Eye
					glEnable(GL_MULTISAMPLE);
					glBindFramebuffer(GL_FRAMEBUFFER, leftEyeDesc.renderFramebuffer);
					glViewport(0, 0, hmd->frameScale.w, hmd->frameScale.h);
					glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
					transBuffer.data = &vrTrans->leftEye;
					transUniform.refreshData();
					glDrawArrays(GL_POINTS, 0, particlesArray.buffer->data->size() / sizeof(Particles::Particle));

					glDisable(GL_MULTISAMPLE);
					leftEyeDesc.copyRenderBuffer();

					vr::Texture_t leftEyeTexture = { (void*)(uintptr_t)leftEyeDesc.resolveTexture,
						vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
					vr::VRCompositor()->Submit(vr::Eye_Left, &leftEyeTexture);

					// Right Eye
					glEnable(GL_MULTISAMPLE);
					glBindFramebuffer(GL_FRAMEBUFFER, rightEyeDesc.renderFramebuffer);
					glViewport(0, 0, hmd->frameScale.w, hmd->frameScale.h);
					glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
					transBuffer.data = &vrTrans->rightEye;
					transUniform.refreshData();
					glDrawArrays(GL_POINTS, 0, particlesArray.buffer->data->size() / sizeof(Particles::Particle));
					glBindFramebuffer(GL_FRAMEBUFFER, 0);

					glDisable(GL_MULTISAMPLE);
					rightEyeDesc.copyRenderBuffer();

					vr::Texture_t rightEyeTexture = { (void*)(uintptr_t)rightEyeDesc.resolveTexture,
						vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
					vr::VRCompositor()->Submit(vr::Eye_Right, &rightEyeTexture);
					glFlush();

					glViewport(0, 0, windowSize.w, windowSize.h);
					glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
					glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
					transBuffer.data = &trans->bufferData;
					transUniform.refreshData();
					glDrawArrays(GL_POINTS, 0, particlesArray.buffer->data->size() / sizeof(Particles::Particle));
				}
			};

			SourceManager sm;
			Particles particles;
			ParticlesData particlesData;
			Buffer particlesBuffer;
			VRDevice hmd;
			Trans vrTrans;
			Transform trans;
			Renderer renderer;
			CUDA::Buffer particlesBufferCUDA;
			NBodyCUDA_Glue glue;


			NBodyCUDAVR(unsigned int _blocks, bool _experiment)
				:
				sm(),
				particles(_blocks * 1024),
				particlesData(&particles),
				particlesBuffer(&particlesData),
				hmd(false),
				vrTrans(&hmd, { 0.01, 30 }),
				trans({ {80.0,0.01,30},{0.01,0.8,0.001},{0.01},500.0 }),
				renderer(&sm, &particlesBuffer, &trans, &vrTrans, &hmd),
				particlesBufferCUDA(CUDA::Buffer::GLinterop),
				glue(_blocks, 0.00001f, 0.001f)
			{
				if (_experiment)
					particles.experimentGalaxy();
				else
					particles.randomGalaxyOptimized();
			}
			void experiment()
			{
				CUDA::Buffer expBuffer(CUDA::Buffer::Device);
				expBuffer.resize(particles.particles.length * sizeof(ExpData));
				glue.experiment((ExpData*)expBuffer.device);
				expBuffer.moveToHost();
				File file("./");
				String<char> answer;
				for (int c0(0); c0 < particles.particles.length; ++c0)
				{
					char tp[50];
					ExpData& data(((ExpData*)expBuffer.host)[c0]);
					sprintf(tp, "%f %f\n", data.r, data.force);
					answer += tp;
				}
				file.createText("answer.txt", answer);
			}
			virtual void init(FrameScale const& _size)override
			{
				glPointSize(1);
				glEnable(GL_DEPTH_TEST);
				trans.init(_size);
				renderer.windowSize = _size;
				renderer.transUniform.dataInit();
				renderer.particlesArray.dataInit();
				particlesBufferCUDA.resize(renderer.particlesArray.buffer->buffer);
				glue.particles = (NBodyCUDAParticle*)particlesBufferCUDA.map();
				vrTrans.leftEye.printInfo();
				vrTrans.rightEye.printInfo();
			}
			virtual void run()override
			{
				vrTrans.update();
				trans.operate();
				if (trans.updated)
				{
					renderer.transUniform.refreshData();
					trans.updated = false;
				}
				renderer.use();
				renderer.run();
				//glFinish();
				glue.run();
			}
			virtual void frameSize(int _w, int _h) override
			{
				trans.resize(_w, _h);
				renderer.windowSize = { _w,_h };
			}
			virtual void framePos(int, int) override
			{
			}
			virtual void frameFocus(int) override
			{
			}
			virtual void mouseButton(int _button, int _action, int _mods) override
			{
				switch (_button)
				{
				case GLFW_MOUSE_BUTTON_LEFT:trans.mouse.refreshButton(0, _action); break;
				case GLFW_MOUSE_BUTTON_MIDDLE:trans.mouse.refreshButton(1, _action); break;
				case GLFW_MOUSE_BUTTON_RIGHT:trans.mouse.refreshButton(2, _action); break;
				}
			}
			virtual void mousePos(double _x, double _y) override
			{
				trans.mouse.refreshPos(_x, _y);
			}
			virtual void mouseScroll(double _x, double _y)override
			{
				if (_y != 0.0)
					trans.scroll.refresh(_y);
			}
			virtual void key(GLFWwindow* _window, int _key, int _scancode, int _action, int _mods) override
			{
				switch (_key)
				{
				case GLFW_KEY_ESCAPE:
					if (_action == GLFW_PRESS)
					{
						particlesBufferCUDA.unmap();
						glfwSetWindowShouldClose(_window, true);
					}
					break;
				case GLFW_KEY_A:trans.key.refresh(0, _action); break;
				case GLFW_KEY_D:trans.key.refresh(1, _action); break;
				case GLFW_KEY_W:trans.key.refresh(2, _action); break;
				case GLFW_KEY_S:trans.key.refresh(3, _action); break;
	}
			}
		};
	}
#endif
#endif
}
