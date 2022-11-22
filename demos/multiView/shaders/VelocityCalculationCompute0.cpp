#version 450 core
layout(local_size_x = 1024)in;

struct Particle
{
	vec3 position;
	float mass;
	vec3 velocity;
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
void main()
{
	uint c0 = 0;
	vec3 r = particles[gl_GlobalInvocationID.x].position;
	vec3 dv = vec3(0);
	for (; c0 < gl_GlobalInvocationID.x; ++c0)
	{
		vec3 dr = particles[c0].position - r;
		dv += (particles[c0].mass / (pow(dot(dr, dr), 1.5) + 0.00001)) * dr;
	}
	for (++c0; c0 < num; ++c0)
	{
		vec3 dr = particles[c0].position - r;
		dv += (particles[c0].mass / (pow(dot(dr, dr), 1.5) + 0.00001)) * dr;
	}
	particles[gl_GlobalInvocationID.x].velocity += dv * G * dt;
}