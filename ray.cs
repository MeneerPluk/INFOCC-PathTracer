using System;
using System.Numerics;

namespace Template {

class Ray
{
	public Ray()
	{
		objIdx = -1;
		inside = false;
	}
	public Ray( Vector3 origin, Vector3 direction, float distance )
	{
		O = origin;
		D = direction;
		t = distance;
		objIdx = -1;
		inside = false;
	}
	public Vector3 O, D;
	public float t;
	public Vector3 N;
	public int objIdx;
	public bool inside;
}
class Ray4
{
	Ray GetRay( int index )
	{
		Vector3 O = new Vector3( Ox4[index], Oy4[index], Oz4[index] );
		Vector3 D = new Vector3( Dx4[index], Dy4[index], Dz4[index] );
		return new Ray( O, D, t4[index] );
	}
	public Vector<float> Ox4, Oy4, Oz4;
	public Vector<float> Dx4, Dy4, Dz4;
	public Vector<float> t4;
	public Vector<float> Nx4, Ny4, Nz4;
	public Vector<int> objIdx;
	public Vector<int> inside;
}

} // namespace Template