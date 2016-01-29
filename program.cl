#define PI 3.14159265359f
#define INVPI (1.0f / 3.14159265359f)
#define MAXDEPTH 20
#define EPSILON 0.0001f

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
	float3 Normal;
	float3 DiffColor;
	float distance;
	int objIdx;
	bool inside;
} Ray;

typedef struct Sphere
{
	float3 Position;
	float radius;
} Sphere;

typedef struct Material
{
	float refl;
	float refr;
	bool emissive;
	float3 Diffuse;
} Material;

float3 toFloat3(Vector3 vec)
{
	float3 ret;
	ret.x = vec.X;
	ret.y = vec.Y;
	ret.z = vec.Z;
	return ret;
}

Vector3 AddFloat3ToVector3(Vector3 v, float3 f)
{
	v.X += f.x;
	v.Y += f.y;
	v.Z += f.z;
	return v;
}

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

int Vector3ToIntRGB(Vector3 color)
{
	int r = min(255, (int) (256.0f * 1.5f * sqrt(color.X)));
	int g = min(255, (int) (256.0f * 1.5f * sqrt(color.Y)));
	int b = min(255, (int) (256.0f * 1.5f * sqrt(color.Z)));
	return (r << 16) + (g << 8) + b;
}

float3 DiffuseReflection(__global int* seed, float3 Normal)
{
	float r1 = Rand(seed);
	float r2 = Rand(seed);
	float r = sqrt(1.0f - r1 * r1);
	float phi = 2 * PI * r2;
	float3 ret;
	ret.x = cos(phi) * r;
	ret.y = sin(phi) * r;
	ret.z = r1;
	if(dot(Normal, ret) < 0) ret *= -1.0f;
	return ret;
}

float3 Refract(Ray r, __global int* seed)
{
	float3 refr;
	float nc = r.inside ? 1 : 1.2f;
	float nt = r.inside ? 1.2f : 1;
	float nnt = nt / nc;
	float ddn = dot(r.Direction, r.Normal);
	float cos2t = 1.0f - nnt * nnt * (1 - ddn * ddn);
	refr = normalize(r.Direction - 2.0 * dot(r.Normal, r.Direction) * r.Normal);

	if(cos2t >= 0)
	{
		float r1 = Rand(seed);
		float a = nt - nc;
		float b = nt + nc;
		float R0 = a * a / (b * b);
		float c = 1 + ddn;
		float Tr = 1 - (R0 + (1 - R0) * c * c * c * c * c);
		if (r1 < Tr) refr = (r.Direction * nnt - r.Normal * (ddn * nnt + (float) sqrt(cos2t)));
	}

	return refr;
}

Material GetMaterial(int objIdx, float3 I)
{
	Material mat;
	if (objIdx == 0)
	{
		mat.refl = 0;
		mat.refr = 0;
		mat.emissive = false;
		int tx = ((int)(I.x * 3.0f + 1000) + (int)(I.z * 3.0f + 1000)) & 1;
		mat.Diffuse = (1,1,1) * ((tx == 1) ? 1.0f : 0.2f);
	}
	if ((objIdx == 1) || (objIdx > 8)) { mat.refl = 0; mat.refr = 0; mat.emissive = false; mat.Diffuse = (1,1,1); }
	if (objIdx == 2) { mat.refl = 0.8f; mat.refr = 0; mat.emissive = false; mat.Diffuse = ( 1, 0.2f, 0.2f ); }
	if (objIdx == 3) { mat.refl = 0; mat.refr = 1; mat.emissive = false; mat.Diffuse = ( 0.9f, 1.0f, 0.9f ); }
	if (objIdx == 4) { mat.refl = 0.8f; mat.refr = 0; mat.emissive = false; mat.Diffuse = ( 0.2f, 0.2f, 1 ); }
	if ((objIdx > 4) && (objIdx < 8)) { mat.refl = 0; mat.refr = 0; mat.emissive = false; mat.Diffuse = (1,1,1); }
	if (objIdx == 8) { mat.refl = 0; mat.refr = 0; mat.emissive = true; mat.Diffuse = (8.5f, 8.5f, 7.0f); }

	return mat;
}

Sphere GetSphere(int index, __global Vector3* origins, __global float* radius)
{
	Sphere s;
	s.Position = toFloat3(origins[index]);
	s.radius = radius[index];
	return s;
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
	float3 out;
	out.x = skybox[idx * 3 + 0];
	out.y = skybox[idx * 3 + 1];
	out.z = skybox[idx * 3 + 2]; 
	return out;
}

Ray IntersectSphere(int idx, Sphere sphere, Ray ray)
{
	float3 L = sphere.Position - ray.Origin;
	float tca = dot(L, ray.Direction);
	if (tca < 0) return ray;
	float d2 = dot(L, L) - tca * tca;
	if (d2 > sphere.radius) return ray;
	float thc = (float)sqrt(sphere.radius - d2);
	float t0 = tca - thc;
	float t1 = tca + thc;
	if (t0 > 0)
	{
		if (t0 > ray.distance) return ray;
		ray.Normal = normalize(ray.Origin + t0 * ray.Direction - sphere.Position);
		ray.objIdx = idx;
		ray.distance = t0;
	}
	else
	{
		if ((t1 > ray.distance) || (t1 < 0)) return ray;
		ray.Normal = normalize(sphere.Position - (ray.Origin + t1 * ray.Direction));
		ray.objIdx = idx;
		ray.distance = t1;
	}

	return ray;
}

Ray Intersect(Ray r, __global Vector3* origins, __global float* radius)
{
	r = IntersectSphere(0, GetSphere(6, origins, radius), r);
	r = IntersectSphere(1, GetSphere(7, origins, radius), r);
	for (int i = 0; i < 6; i++) r = IntersectSphere(i+2, GetSphere(i, origins, radius), r);
	r = IntersectSphere(8,  GetSphere(8, origins, radius), r);

	return r;
}

float3 Sample(Ray r, __global Vector3* origins, __global float* radius, __global float* skybox, __global int* seed)
{
    int depth = 0;
	r.DiffColor = (1, 1, 1);
	while(depth < MAXDEPTH)
	{
		r = Intersect(r, origins, radius);
		if (r.objIdx == -1) return SampleSkyBox(r.Direction, skybox);

		float3 I = r.Origin + r.distance * r.Direction;
		Material m = GetMaterial(r.objIdx, I);
		if(m.emissive){ return m.Diffuse;}
	
		float r0 = Rand(seed);
		float3 refr = (0, 0, 0);
		if (r0 < m.refr)
		{
			refr = Refract(r, seed);
			Ray exray;
			exray.Origin = I + refr * EPSILON;
			exray.Direction = refr;
			exray.distance = 1e34f;
			exray.inside = (dot(r.Normal, refr) < 0);
			exray.DiffColor = m.Diffuse * r.DiffColor;
			r = exray;
			depth++;
		}
		else if (r0 < (m.refl + m.refr))
		{
			refr = normalize(r.Direction - 2.0 * dot(r.Normal, r.Direction) * r.Normal);
			Ray exray;
			exray.Origin = I + refr * EPSILON;
			exray.Direction = refr;
			exray.distance = 1e34f;
			exray.DiffColor = m.Diffuse * r.DiffColor;
			r = exray;
			depth++;
		}
		else
		{
			refr = DiffuseReflection(seed, r.Normal);
			Ray exray;
			exray.Origin = I + refr * EPSILON;
			exray.Direction = refr;
			exray.distance = 1e34f;
			exray.DiffColor = dot(refr, r.Normal) * m.Diffuse * r.DiffColor;
			r = exray;
			depth++;
		}
	}
	float3 out;
	out.x = r.DiffColor.x;
	out.y = r.DiffColor.y;
	out.z = r.DiffColor.z;
	return out;
}

__kernel void device_function( Vector3 p1, Vector3 p2, Vector3 p3, Vector3 up, Vector3 right, Vector3 pos, float lensSize, float w, float h, __global int* seed, __global int* screen, __global float* skybox, __global Vector3* origins, __global float* radius, __global Vector3* acc)
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

	Ray r;
	r.Origin = P;
	r.Direction = D;
	r.distance = 1e34f;
	r.objIdx = -1;

	float3 sample = Sample(r, origins, radius, skybox, seed);
	acc[get_global_id(0)] = AddFloat3ToVector3(acc[get_global_id(0)], sample);
	screen[get_global_id(0)] = Vector3ToIntRGB(acc[get_global_id(0)]);
}