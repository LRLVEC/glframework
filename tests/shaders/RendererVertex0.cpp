#version 450 core
layout(std140, row_major, binding = 0)uniform transBuffer
{
	mat4 trans;
};
layout(location = 0)in vec3 position;
layout(location = 1)in vec3 velocity;
out vec4 fragColor;
void main()
{
	gl_Position = trans * vec4(position, 1);
	float k = tanh(length(velocity) / 2.5);
	fragColor = vec4(1 - k, k, k, 1);
}