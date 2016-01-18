using System;
using System.Numerics;

namespace Template {

class Material
{
	public float refl;
	public float refr;
	public bool emissive;
	public Vector3 diffuse;
}
class Scene
{
	float [] skybox;
	const float LIGHTSIZE = 0.3f;
	const float LIGHTSCALE = 1.0f;
	public static Vector3 lightColor = new Vector3( 8.5f * LIGHTSCALE, 8.5f * LIGHTSCALE, 7.0f * LIGHTSCALE );
	static Sphere plane1, plane2; // very large spheres approximating planes
	static Sphere [] sphere = new Sphere[6];
	static Sphere light;
	const float PI = 3.14159265359f;
	const float INVPI = 1.0f / PI;
	public Scene()
	{
		// define geometry
		plane1 = new Sphere( 0.0f, -4999.0f, 0.0f, 4998.5f * 4998.5f );		// bottom plane
		plane2 = new Sphere( 0.0f, 0.0f, -5000.0f, 4993.0f * 4993f );		// back plane
		for( int i = 0; i < 3; i++ )
		{
			sphere[i] = new Sphere( -0.8f + i * 0.8f, 0, -2, 0.3f * 0.3f );
			sphere[i + 3] = new Sphere( -0.8f + i * 0.8f, -0.8f, -2, 0.5f * 0.5f );
		}
		light = new Sphere( 2.7f, 1.7f, -0.5f, LIGHTSIZE * LIGHTSIZE );
		// load skybox
		skybox = new float[2500 * 1250 * 3];
		WinFileIO wf = new WinFileIO( skybox );
		wf.OpenForReading( "../../assets/sky_15.raw" );
		wf.Read( 2500 * 650 * 4 * 3 );
		for( int i = 2500 * 650 * 3; i < (2500 * 1250 * 3); i++ ) skybox[i] = 0;
		wf.Dispose();
	}
	public Vector3 SampleSkydome( Vector3 D )
	{
		int u = (int)(2500.0f * 0.5f * (1.0f + Math.Atan2( D.X, -D.Z ) * INVPI));
		int v = (int)(1250.0f * Math.Acos( D.Y ) * INVPI);
		int idx = u + v * 2500;
		return new Vector3( skybox[idx * 3 + 0], skybox[idx * 3 + 1], skybox[idx * 3 + 2] );
	}
	private static void IntersectSphere( int idx, Sphere sphere, Ray ray )
	{
		Vector3 L = sphere.pos - ray.O;
		float tca = Vector3.Dot( L, ray.D );
		if (tca < 0) return;
		float d2 = Vector3.Dot( L, L ) - tca * tca;
		if (d2 > sphere.r) return;
		float thc = (float)Math.Sqrt( sphere.r - d2 );
		float t0 = tca - thc;
		float t1 = tca + thc;
		if (t0 > 0)
		{
			if (t0 > ray.t) return;
			ray.N = Vector3.Normalize( ray.O + t0 * ray.D - sphere.pos );
			ray.objIdx = idx;
			ray.t = t0;
		}
		else
		{
			if ((t1 > ray.t) || (t1 < 0)) return;
			ray.N = Vector3.Normalize( sphere.pos - (ray.O + t1 * ray.D) );
			ray.objIdx = idx;
			ray.t = t1;
		}
	}
	public Material GetMaterial( int objIdx, Vector3 I )
	{
		Material mat = new Material();
		if (objIdx == 0) 
		{
			// procedural checkerboard pattern for floor plane
			mat.refl = mat.refr = 0;
			mat.emissive = false;
			int tx = ((int)(I.X * 3.0f + 1000) + (int)(I.Z * 3.0f + 1000)) & 1;
			mat.diffuse = Vector3.One * ((tx == 1) ? 1.0f : 0.2f);
		}
		if ((objIdx == 1) || (objIdx > 8)) { mat.refl = mat.refr = 0; mat.emissive = false; mat.diffuse = Vector3.One; }
		if (objIdx == 2) { mat.refl = 0.8f; mat.refr = 0; mat.emissive = false; mat.diffuse = new Vector3( 1, 0.2f, 0.2f ); }
		if (objIdx == 3) { mat.refl = 0; mat.refr = 1; mat.emissive = false; mat.diffuse = new Vector3( 0.9f, 1.0f, 0.9f ); }
		if (objIdx == 4) { mat.refl = 0.8f; mat.refr = 0; mat.emissive = false; mat.diffuse = new Vector3( 0.2f, 0.2f, 1 ); }
		if ((objIdx > 4) && (objIdx < 8)) { mat.refl = mat.refr = 0; mat.emissive = false; mat.diffuse = Vector3.One; }
		if (objIdx == 8) { mat.refl = mat.refr = 0; mat.emissive = true; mat.diffuse = lightColor; }
		return mat;
	}
	public static void Intersect( Ray ray )
	{
		IntersectSphere( 0, plane1, ray );
		IntersectSphere( 1, plane2, ray );
		for( int i = 0; i < 6; i++ ) IntersectSphere( i + 2, sphere[i], ray );
		IntersectSphere( 8, light, ray );
	}
}

} // namespace Template
