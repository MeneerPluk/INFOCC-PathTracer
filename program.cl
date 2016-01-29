#define INVPI (1.0f / 3.14159265359f)

typedef struct Vector3
{
	float X;
	float Y;
	float Z;
} Vector3;

typedef struct Ray
{
	float3 Origin;
	float3 Direction;
	float distance;
	int objIdx;
	bool inside;
} Ray;

int Xor(int seed)
{
	int t = seed ^ (seed << 11);
	seed = (seed ^ (seed >> 19) ^ (t ^ ( t >> 8)));
	return seed;
}

float Rand(__global int* rng)
{
	int seed = rng[get_global_id(0)];
	seed = Xor(seed);
	rng[get_global_id(0)] = seed;
	return ((float)seed / 2147483647);
}

float3 toFloat3(Vector3 vec)
{
	float3 ret;
	ret.x = vec.X;
	ret.y = vec.Y;
	ret.z = vec.Z;
	return ret;
}

Vector3 toVector3(float3 f)
{
	Vector3 vec;
	vec.X = f.x;
	vec.Y = f.y;
	vec.Z = f.z;
	return vec;
}

float3 SampleSkyBox(float3 Dir, __global float* skybox)
{
	int u = (int)(2500.0f * 0.5f * (1.0f + atan2(Dir.x, -Dir.z) * INVPI));
	int v = (int)(1250.0f * acos(Dir.y) * INVPI);
	int idx = u + (v * 2500);
	return (skybox[idx * 3 + 0], skybox[idx * 3 + 1], skybox[idx * 3 + 2]);
}

void Intersect(Ray r)
{
	IntersectSphere(0, plane1, r);
	IntersectSphere(1, plane2, r);
	for (int i = 0; i < 6; i++) IntersectSphere(i+2, sphere[i], r);
    IntersectSphere(8, light, r);
}

float3 Sample(Ray r, int depth)
{
 return (0,0,0);
}

__kernel void device_function( Vector3 p1, Vector3 p2, Vector3 p3, Vector3 up, Vector3 right, Vector3 pos, float lensSize, float w, float h, __global int* seed, __global Vector3* screen, __global float* skybox)
{
	float3 fp1 = toFloat3(p1);
	float3 fp2 = toFloat3(p2);
	float3 fp3 = toFloat3(p3);
	float3 fup = toFloat3(up);
	float3 fright = toFloat3(right);
	float3 fpos = toFloat3(pos);

	float x = floor(fmod(get_global_id(0), w));
	float y = floor(get_global_id(0) / w);
	
	float r0 = Rand(seed);
	float r1 = Rand(seed);
	float r2 = Rand(seed) - 0.5f;
	float r3 = Rand(seed) - 0.5f;

	float u = (x + r0) / w;
	float v = (y + r1) / w;
	
	float3 T = fp1 + u * (fp2 - fp1) + v * (fp3 - fp1);
	float3 P = fpos + lensSize * (r2 * fright + r3 * fup);
	float3 D = normalize(T - P);

	float3 skyboxsample = 1.0f * SampleSkyBox(D, skybox);
	screen[get_global_id(0)] = toVector3(skyboxsample);
}