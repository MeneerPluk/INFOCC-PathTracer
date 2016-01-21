using System;
using System.Diagnostics;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;

// making vectors work in VS2013:
// - Uninstall Nuget package manager
// - Install Nuget v2.8.6 or later
// - In Package Manager Console: Install-Package System.Numerics.Vectors -Pre

namespace Template
{

    class Game
    {
        // member variables
        public Surface screen;                  // target canvas
        Camera camera;                          // camera
        Scene scene;                            // hardcoded scene
        Stopwatch timer = new Stopwatch();      // timer
        Vector3[] accumulator;                  // buffer for accumulated samples
        int spp = 0;                            // samples per pixel; accumulator will be divided by this
        int runningTime = -1;                   // running time (from commandline); default = -1 (infinite)
        bool useGPU = true;                     // GPU code enabled (from commandline)
        int gpuPlatform = 0;                    // OpenCL platform to use (from commandline)
        bool firstFrame = true;                 // first frame: used to start timer once
                                                // constants for rendering algorithm
        const float PI = 3.14159265359f;
        const float INVPI = 1.0f / PI;
        const float EPSILON = 0.0001f;
        const int MAXDEPTH = 20;

        // clear the accumulator: happens when camera moves
        private void ClearAccumulator()
        {
            for (int s = screen.width * screen.height, i = 0; i < s; i++)
                accumulator[i] = Vector3.Zero;
            spp = 0;
        }
        // initialize renderer: takes in command line parameters passed by template code
        public void Init(int rt, bool gpu, int platformIdx)
        {
            // pass command line parameters
            runningTime = rt;
            useGPU = gpu;
            gpuPlatform = platformIdx;
            // initialize accumulator
            accumulator = new Vector3[screen.width * screen.height];
            ClearAccumulator();
            // setup scene
            scene = new Scene();
            // setup camera
            camera = new Camera(screen.width, screen.height);
        }
        // sample: samples a single path up to a maximum depth
        private Vector3 Sample(Ray ray, int depth)
        {
            // find nearest ray/scene intersection
            Scene.Intersect(ray);
            if (ray.objIdx == -1)
            {
                // no scene primitive encountered; skybox
                return 1.0f * scene.SampleSkydome(ray.D);
            }
            // calculate intersection point
            Vector3 I = ray.O + ray.t * ray.D;
            // get material at intersection point
            Material material = scene.GetMaterial(ray.objIdx, I);
            if (material.emissive)
            {
                // hit light
                return material.diffuse;
            }
            // terminate if path is too long
            if (depth >= MAXDEPTH) return Vector3.Zero;
            // handle material interaction
            float r0 = RTTools.RandomFloat();
            Vector3 R = Vector3.Zero;
            if (r0 < material.refr)
            {
                // dielectric: refract or reflect
                RTTools.Refraction(ray.inside, ray.D, ray.N, ref R);
                Ray extensionRay = new Ray(I + R * EPSILON, R, 1e34f);
                extensionRay.inside = (Vector3.Dot(ray.N, R) < 0);
                return material.diffuse * Sample(extensionRay, depth + 1);
            }
            else if ((r0 < (material.refl + material.refr)) && (depth < MAXDEPTH))
            {
                // pure specular reflection
                R = Vector3.Reflect(ray.D, ray.N);
                Ray extensionRay = new Ray(I + R * EPSILON, R, 1e34f);
                return material.diffuse * Sample(extensionRay, depth + 1);
            }
            else
            {
                // diffuse reflection
                R = RTTools.DiffuseReflection(RTTools.GetRNG(), ray.N);
                Ray extensionRay = new Ray(I + R * EPSILON, R, 1e34f);
                return Vector3.Dot(R, ray.N) * material.diffuse * Sample(extensionRay, depth + 1);
            }
        }
        // tick: renders one frame
        public void Tick()
        {
            // initialize timer
            if (firstFrame)
            {
                timer.Reset();
                timer.Start();
                firstFrame = false;
            }
            // handle keys, only when running time set to -1 (infinite)
            if (runningTime == -1) if (camera.HandleInput())
                {
                    // camera moved; restart
                    ClearAccumulator();
                }
            // render
            if (false) // if (useGPU)
            {
                // add your CPU + OpenCL path here
                // mind the gpuPlatform parameter! This allows us to specify the platform on our
                // test system.
                // note: it is possible that the automated test tool provides you with a different
                // platform number than required by your hardware. In that case, you can hardcode
                // the platform during testing (ignoring gpuPlatform); do not forget to put back
                // gpuPlatform before submitting!
            }
            else
            {
                // this is your CPU only path
                float scale = 1.0f / (float)++spp;

                Parallel.For(0, screen.height, y =>
                {
                    Random r = RTTools.GetRNG();
                    for (int x = 0; x < screen.width; x++)
                    {
                        // generate primary ray
                        Ray ray = camera.Generate(r, x, y);
                        // trace path
                        int pixelIdx = x + y * screen.width;
                        accumulator[pixelIdx] += Sample(ray, 0);
                        // plot final color
                        screen.pixels[pixelIdx] = RTTools.Vector3ToIntegerRGB(scale * accumulator[pixelIdx]);
                    }
                });
            }
            // stop and report when max render time elapsed
            int elapsedSeconds = (int)(timer.ElapsedMilliseconds / 1000);
            if (runningTime != -1) if (elapsedSeconds >= runningTime)
                {
                    OpenTKApp.Report((int)timer.ElapsedMilliseconds, spp, screen);
                }
        }
    }

} // namespace Template