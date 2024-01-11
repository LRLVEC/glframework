#version 450 core
layout(local_size_x = 1024)in;

struct Particle
{
	vec3 position;
	float mass;
	vec3 velocity;
};
struct ParticleShared
{
	vec3 position;
	float mass;
};
layout(std430, binding = 1)buffer ParticlesBuffer
{
	Particle particles[];
};
layout(std140, binding = 3)uniform ParameterBuffer
{
	float dt;
	float G;
	uint num;
};

shared ParticleShared shared_particles[1024];

void main()
{
	uint c0 = 0;
	vec3 r = particles[gl_GlobalInvocationID.x].position;
	vec3 dv = vec3(0);
	for (; c0 < num; c0 += 1024)
	{
		shared_particles[gl_LocalInvocationID.x].position = particles[c0 + gl_LocalInvocationID.x].position;
		shared_particles[gl_LocalInvocationID.x].mass = particles[c0 + gl_LocalInvocationID.x].mass;
		barrier();
		for (int c1 = 0; c1 < 1024; c1++)
		{
			vec3 dr = shared_particles[c1].position - r;
			float drr = inversesqrt(clamp(dot(dr, dr), 0.0001, 10000000.0));
			drr = drr * drr * drr;
			dv += (shared_particles[c1].mass * drr) * dr;
			// dv += (particles[c0].mass / (pow(dot(dr, dr), 1.5) + 0.00001)) * dr;
		}
		barrier();
	}
	particles[gl_GlobalInvocationID.x].velocity += dv * G * dt;
}