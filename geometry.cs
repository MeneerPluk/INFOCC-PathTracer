using System;
using System.Numerics;

namespace Template
{
class Plane
{
	public Plane( float A, float B, float C, float D )
	{
		N.X = A;
		N.Y = B;
		N.Z = C;
		d = D;
	}
	public Plane( Vector3 ABC, float D )
	{
		N = ABC;
		d = D;
	}
	public Vector3 N;
	public float d;
}
class Sphere
{
	public Sphere( float x, float y, float z, float radius )
	{
		pos.X = x;
		pos.Y = y;
		pos.Z = z;
		r = radius;
	}
	public Vector3 pos;
	public float r;
}

} // namespace Template
