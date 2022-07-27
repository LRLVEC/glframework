#pragma once
#include <GL/OPENVR/openvr.h>
#include <GL/_OpenGL.h>
#pragma comment(lib, "GL/openvr_api.lib")

#include <_String.h>
#include <_Math.h>

namespace OpenGL
{
	namespace VR
	{
		struct FrameBufferDesc
		{
			GLuint depthBuffer;
			GLuint renderTexture;
			GLuint renderFramebuffer;
			GLuint resolveTexture;
			GLuint resolveFramebuffer;
			FrameScale size;

			FrameBufferDesc(FrameScale _size)
				:
				size(_size)
			{
				glGenFramebuffers(1, &renderFramebuffer);
				glBindFramebuffer(GL_FRAMEBUFFER, renderFramebuffer);

				glGenRenderbuffers(1, &depthBuffer);
				glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
				glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH_COMPONENT, _size.w, _size.h);
				glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);

				glGenTextures(1, &renderTexture);
				glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, renderTexture);
				glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA8, _size.w, _size.h, true);
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, renderTexture, 0);

				glGenFramebuffers(1, &resolveFramebuffer);
				glBindFramebuffer(GL_FRAMEBUFFER, resolveFramebuffer);

				glGenTextures(1, &resolveTexture);
				glBindTexture(GL_TEXTURE_2D, resolveTexture);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, _size.w, _size.h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, resolveTexture, 0);

				glBindFramebuffer(GL_FRAMEBUFFER, 0);

				// check FBO status
				//GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
				//if (status != GL_FRAMEBUFFER_COMPLETE)return false;
				//glBindFramebuffer(GL_FRAMEBUFFER, 0);
				//return true;
			}
			void copyRenderBuffer()
			{
				glBindFramebuffer(GL_READ_FRAMEBUFFER, renderFramebuffer);
				glBindFramebuffer(GL_DRAW_FRAMEBUFFER, resolveFramebuffer);
				glBindTexture(GL_TEXTURE_2D, resolveTexture);
				glBlitFramebuffer(0, 0, size.w, size.h, 0, 0, size.w, size.h, GL_COLOR_BUFFER_BIT, GL_LINEAR);
				glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
				glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
			}
		};
		struct Object
		{
			unsigned int numDevice;
			String<char> name;
			vr::ETrackedDeviceClass objClass;
			Math::mat4<float> pos;		//trans: world space to head space in rasterization or head space to world space in ray tracing, offset is in head space...
			Math::vec3<float> velocity;	//believed in world space...
			Math::vec3<float> omega;	//believed in world space...
			vr::ETrackingResult trackingResult;
			bool posValid;
			bool connected;

			Object()
			{
			}
			Object(unsigned int _numDevice, vr::ETrackedDeviceClass _objClass, vr::TrackedDevicePose_t const& a)
				:
				numDevice(_numDevice),
				objClass(_objClass),
				trackingResult(vr::TrackingResult_Uninitialized),
				posValid(false),
				connected(false)
			{
				switch (_objClass)
				{
					case vr::TrackedDeviceClass_Controller:			name = "Controller"; break;
					case vr::TrackedDeviceClass_HMD:				name = "HMD"; break;
					case vr::TrackedDeviceClass_Invalid:			name = "Invalid"; break;
					case vr::TrackedDeviceClass_GenericTracker:		name = "Generic Tracker"; break;
					case vr::TrackedDeviceClass_TrackingReference:	name = "Tracking Reference"; break;
					default:										name = "???"; break;
				}
				char t[20];
				sprintf(t, "_%u", _numDevice);
				name += t;
				update(a);
			}
			void update(vr::TrackedDevicePose_t const& a)
			{
				Math::mat<float, 3, 4>m0(*(Math::mat<float, 3, 4>*) & a.mDeviceToAbsoluteTracking);
				Math::mat3<float>m1(m0);
				pos = !m1;
				Math::vec4<float>dr(-m0.column(3));
				dr.data[3] = 0;
				dr = (pos, dr);
				dr.data[3] = 1;
				pos.setCol(dr, 3);
				//new(&pos)Math::mat4<float>(*(Math::mat<float, 3, 4>*) & a.mDeviceToAbsoluteTracking);
				//pos.rowVec[3] = { 0,0,0,1.0f };
				velocity = *(Math::vec3<float>*) & a.vVelocity;
				omega = *(Math::vec3<float>*) & a.vAngularVelocity;
				trackingResult = a.eTrackingResult;
				posValid = a.bPoseIsValid;
				connected = a.bDeviceIsConnected;
			}
			void updateOptiX(vr::TrackedDevicePose_t const& a);
			void printInfo()const
			{
				name.print();
				::printf(":");
				if (posValid && connected)
				{
					pos.printInfo("\nPos: ");
					velocity.printInfo("Velocity: ");
					omega.printInfo("\nOmega: ");
					::printf("\n");
				}
				else
				{
					::printf("Invalid!\n");
				}
			}
		};
		struct VRDevice
		{
			vr::IVRSystem* hmd;
			FrameScale frameScale;
			Object objects[vr::k_unMaxTrackedDeviceCount];
			Vector<unsigned int>validObjects;

			VRDevice(bool useRecommendSize)
				:
				hmd(nullptr),
				frameScale({ -1,-1 })
			{
				vr::EVRInitError error = vr::VRInitError_None;
				hmd = vr::VR_Init(&error, vr::VRApplication_Scene);
				if (error != vr::VRInitError_None)
				{
					hmd = nullptr;
					char buf[1024];
					printf("Unable to init VR runtime: %s", vr::VR_GetVRInitErrorAsEnglishDescription(error));
					printf("VR_Init Failed");
				}
				if (hmd)
				{
					if (useRecommendSize)
					{
						unsigned int width, height;
						hmd->GetRecommendedRenderTargetSize(&width, &height);
						frameScale.w = width;
						frameScale.h = height;
					}
					else
					{
						frameScale.w = 1080;
						frameScale.h = 1200;
					}
					getObjects();
				}
			}
			~VRDevice()
			{
				for (unsigned int c0 = 0; c0 < vr::k_unMaxTrackedDeviceCount; ++c0)
					objects[c0].~Object();
			}
			void getObjects()
			{
				if (hmd)
				{
					vr::TrackedDevicePose_t devices[vr::k_unMaxTrackedDeviceCount];
					vr::VRCompositor()->WaitGetPoses(devices, vr::k_unMaxTrackedDeviceCount, NULL, 0);
					validObjects.clear();
					for (unsigned int c0 = 0; c0 < vr::k_unMaxTrackedDeviceCount; ++c0)
					{
						objects[c0].~Object();
						vr::ETrackedDeviceClass objClass = hmd->GetTrackedDeviceClass(c0);
						new(objects + c0)Object(c0, objClass, devices[c0]);
						if (devices[c0].bPoseIsValid && devices[c0].bDeviceIsConnected && objClass)
							validObjects.pushBack(c0);
					}
				}
			}
			void refreshHMD()
			{
				if (hmd)
				{
					vr::TrackedDevicePose_t device;
					vr::VRCompositor()->WaitGetPoses(&device, 1, NULL, 0);
					objects[0].update(device);
				}
			}
			void refreshHMDOptiX()
			{
				if (hmd)
				{
					vr::TrackedDevicePose_t device;
					vr::VRCompositor()->WaitGetPoses(&device, 1, NULL, 0);
					objects[0].updateOptiX(device);
				}
			}
			operator vr::IVRSystem* ()
			{
				return hmd;
			}
			void printInfo()
			{
				::printf("FrameSize: [%d, %d]\n", frameScale.w, frameScale.h);
			}
		};

		struct Trans
		{
			struct Perspective
			{
				double zNear;
				double zFar;
			};
			struct SingleEye :Buffer::Data
			{
				vr::EVREye eye;
				VRDevice* hmd;
				Perspective* persp;
				Math::mat4<float> proj;
				//Normally doesn't change until you change the distance between two eyes.
				//In fact, we can use a vec3<float> instead because it's just an offset from eye to head in head space.
				//Math::mat4<float> offset;
				Math::vec3<float> offset;
				Math::mat4<float> trans;
				Math::mat4<float> answer;

				SingleEye() = delete;
				SingleEye(VRDevice* _hmd, vr::EVREye _eye, Perspective* _persp)
					:
					hmd(_hmd),
					eye(_eye),
					persp(_persp)
				{
					updateProj();
					updateAll();
				}
				void updateProj()
				{
					vr::HmdMatrix44_t projMat4 = hmd->hmd->GetProjectionMatrix(eye, persp->zNear, persp->zFar);
					proj = (*(Math::mat4<float>*) & projMat4);
				}
				void updateOffset()
				{
					vr::HmdMatrix34_t offsetMat4 = hmd->hmd->GetEyeToHeadTransform(eye);
					offset = { offsetMat4.m[0][3],offsetMat4.m[1][3], offsetMat4.m[2][3] };
				}
				void updateTrans(Object const& _hmd)
				{
					trans = _hmd.pos;
					trans.array[0][3] -= offset.data[0];
					trans.array[1][3] -= offset.data[1];
					trans.array[2][3] -= offset.data[2];
				}
				void updateAll()//if proj or offset changes, use this
				{
					updateProj();
					updateOffset();
					updateTrans(hmd->objects[0]);
					answer = (proj, trans);
				}
				void update()
				{
					updateTrans(hmd->objects[0]);
					answer = (proj, trans);
				}
				virtual void* pointer()override
				{
					return (void*)(answer.array);
				}
				virtual unsigned int size()override
				{
					return sizeof(answer);
				}
				void printInfo()const
				{
					::printf(eye == vr::Eye_Left ? "Left eye:\n" : "Right eye:\n");
					proj.printInfo("Proj: ");
					offset.printInfo("Offset: ");
					trans.printInfo("\nTrans: ");
				}
			};
			VRDevice* hmd;
			Perspective persp;
			SingleEye leftEye;
			SingleEye rightEye;

			Trans(VRDevice* _hmd, Perspective _persp)
				:
				hmd(_hmd),
				persp(_persp),
				leftEye(_hmd, vr::Eye_Left, &persp),
				rightEye(_hmd, vr::Eye_Right, &persp)
			{
			}
			void update()
			{
				hmd->refreshHMD();
				leftEye.update();
				rightEye.update();
			}
			void updateAll()
			{
				hmd->refreshHMD();
				leftEye.updateAll();
				rightEye.updateAll();
			}
		};
	}
}