#version 450 core
layout(location = 0) in vec2 position;
out vec2 texCood;
void main()
{
	gl_Position = vec4(position, 0, 1);
	texCood = (position + vec2(1)) / 2;
}