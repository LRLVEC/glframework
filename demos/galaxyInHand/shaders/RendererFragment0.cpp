#version 450 core
in ColorPos
{
	vec4 fragColor;
	vec3 pos;
};
out vec4 Color;
void main()
{
	Color = fragColor;
	// if (any(lessThan(pos, vec3(-2.0))) || any(greaterThan(pos, vec3(2.0))))
	// {
	// 	discard;
	// }
}