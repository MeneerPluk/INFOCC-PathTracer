using System;
using System.Numerics;

namespace Template
{
class RTTools
{
	// random number generators
	private static Random globalRng = new Random(); 
    private static Random rng = null;
	const float PI = 3.14159265359f;
	const float BRIGHTNESS = 1.5f;
	public static float RandomFloat() 
    { 
        Random inst = rng; 
        if (inst == null) 
        { 
            int seed; 
            lock (globalRng) seed = globalRng.Next(); 
            rng = inst = new Random( seed ); 
        } 
        return (float)inst.NextDouble(); 
    }
	public static Random GetRNG()
	{
        Random inst = rng; 
        if (inst == null) 
        { 
            int seed; 
            lock (globalRng) seed = globalRng.Next(); 
            rng = inst = new Random( seed ); 
        } 
		return rng;
	}
	static public Vector3 DiffuseReflection( Random rng, Vector3 N )
	{
		float r1 = (float)rng.NextDouble();
		float r2 = (float)rng.NextDouble();
		float r = (float)Math.Sqrt( 1.0 - r1 * r1 );
		float phi = 2 * PI * r2;
		Vector3 R;
		R.X = (float)Math.Cos( phi ) * r;
		R.Y = (float)Math.Sin( phi ) * r;
		R.Z = r1;
		if (Vector3.Dot( N, R ) < 0) R *= -1.0f;
		return R;
	}
	static public void Refraction( bool inside, Vector3 D, Vector3 N, ref Vector3 R )
	{
		float nc = inside ? 1 : 1.2f, nt = inside ? 1.2f : 1;
		float nnt = nt / nc, ddn = Vector3.Dot( D, N ); 
		float cos2t = 1.0f - nnt * nnt * (1 - ddn * ddn);
		R = Vector3.Reflect( D, N );
		if (cos2t >= 0)
		{
			float r1 = RTTools.RandomFloat();
			float a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 + ddn;
			float Tr = 1 - (R0 + (1 - R0) * c * c * c * c * c);
			if (r1 < Tr) R = (D * nnt - N * (ddn * nnt + (float)Math.Sqrt( cos2t )));
		}
	}
	static public int Vector3ToIntegerRGB( Vector3 color )
	{
		// apply gamma correction and convert to integer rgb
		int r = (int)Math.Min( 255, 256.0f * BRIGHTNESS * Math.Sqrt( color.X ) );
		int g = (int)Math.Min( 255, 256.0f * BRIGHTNESS * Math.Sqrt( color.Y ) );
		int b = (int)Math.Min( 255, 256.0f * BRIGHTNESS * Math.Sqrt( color.Z ) );
		return (r << 16) + (g << 8) + b;
	}
}

} // namespace Template
