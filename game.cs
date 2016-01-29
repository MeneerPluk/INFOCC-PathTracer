using System;
using System.Diagnostics;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using Cloo;
using System.IO;
using System.Resources;
using OpenTK.Audio.OpenAL;

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

        //For tiling:
        int tileCount = 1;                      //TileCount is the amount of tiles in one row or column. It used to be the total amount of tiles,
        int tileWidth;                          //but we had to calculate the square root of that more often than the power of 2 of this.
        int tileHeight;                         //Less calculations is more optimization!

        //OpenCL variables
        ComputeKernel kernel;
        ComputeContext context;
        ComputeCommandQueue queue;
        ComputeBuffer<int> rngBuffer, screenPixels;
        ComputeBuffer<float> skyBox, radiusBuffer;
        ComputeBuffer<Vector3> originBuffer, accBuffer;
        int[] rngSeed;
        long[] workSize;

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
            //Determine tile width and height
            tileCount = GreatestDiv(screen.width, screen.height);
            tileWidth = screen.width/tileCount;
            tileHeight = screen.height/tileCount;
            // initialize accumulator
            accumulator = new Vector3[screen.width * screen.height];
            ClearAccumulator();
            // setup scene
            scene = new Scene();
            // setup camera
            camera = new Camera(screen.width, screen.height);

            //Init OpenCL
            ComputePlatform platform = ComputePlatform.Platforms[gpuPlatform];
            context = new ComputeContext(
                ComputeDeviceTypes.Gpu,
                new ComputeContextPropertyList(platform),
                null,
                IntPtr.Zero
                );
            var streamReader = new StreamReader("../../program.cl");
            string clSource = streamReader.ReadToEnd();
            streamReader.Close();

            ComputeProgram program = new ComputeProgram(context, clSource);

            //try to compile
            try
            {
                program.Build(null, null, null, IntPtr.Zero);
            }
            catch
            {
                Console.Write("error in kernel code:\n");
                Console.Write(program.GetBuildLog(context.Devices[0]) + "\n");
            }
            kernel = program.CreateKernel("device_function");

            //setup RNG
            rngSeed = new int[screen.width * screen.height];
            Random r = RTTools.GetRNG();
            for (int i = 0; i < rngSeed.Length; i++)
                rngSeed[i] = r.Next();

            //import buffers etc to GPU
            Vector3[] data = new Vector3[screen.width * screen.height];
            Vector3[] sphereOrigins = Scene.GetOrigins;
            float[] sphereRadii = Scene.GetRadii;

            var FlagRW = ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer;
            var FlagR = ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer;

            rngBuffer = new ComputeBuffer<int>(context, FlagRW, rngSeed);
            screenPixels = new ComputeBuffer<int>(context, FlagRW, screen.pixels);
            skyBox = new ComputeBuffer<float>(context, FlagR, scene.skybox);
            originBuffer = new ComputeBuffer<Vector3>(context, FlagR, sphereOrigins);
            radiusBuffer = new ComputeBuffer<float>(context, FlagR, sphereRadii);
            accBuffer = new ComputeBuffer<Vector3>(context, FlagRW, accumulator);

            kernel.SetValueArgument(0, camera.p1);
            kernel.SetValueArgument(1, camera.p2);
            kernel.SetValueArgument(2, camera.p3);
            kernel.SetValueArgument(3, camera.up);
            kernel.SetValueArgument(4, camera.right);
            kernel.SetValueArgument(5, camera.pos);
            kernel.SetValueArgument(6, camera.lensSize);
            kernel.SetValueArgument(7, (float)screen.width);
            kernel.SetValueArgument(8, (float)screen.height);
            kernel.SetMemoryArgument(9, rngBuffer);
            kernel.SetMemoryArgument(10, screenPixels);
            kernel.SetMemoryArgument(11, skyBox);
            kernel.SetMemoryArgument(12, originBuffer);
            kernel.SetMemoryArgument(13, radiusBuffer);
            kernel.SetMemoryArgument(14, accBuffer);

            queue = new ComputeCommandQueue(context, context.Devices[0], 0);

            long[] tempWorkSize = { screen.width * screen.height };             //For some reason, doing this directly produces a build error.
            workSize = tempWorkSize;                                            //Luckily, this works.
        }

        int GreatestDiv(int x, int y)
        {
            int remainder;
            while (y != 0)
            {
                remainder = x%y;
                x = y;
                y = remainder;
            }
            return x;
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
                    queue.WriteToBuffer(accumulator, accBuffer, true, null);

                    //Update camera in the OpenCL kernel.
                    kernel.SetValueArgument(0, camera.p1);
                    kernel.SetValueArgument(1, camera.p2);
                    kernel.SetValueArgument(2, camera.p3);
                    kernel.SetValueArgument(3, camera.up);
                    kernel.SetValueArgument(4, camera.right);
                    kernel.SetValueArgument(5, camera.pos);
                }
            // render
            if (useGPU) // if (useGPU)
            {
                //The only value really necessary to pass on to the GPU every step is the scale.
                float scale = 1.0f / (float)++spp;
                kernel.SetValueArgument(15, scale);

                queue.Execute(kernel, null, workSize, null, null);
                queue.Finish();

                //Read the result onto the screen.
                queue.ReadFromBuffer(screenPixels, ref screen.pixels, true, null);
            }
            else
            {
                float scale = 1.0f / (float)++spp;

                //Since the tiles guarantee a certain amount of data locality,
                //we can easily use a Parallel.For over all of them.
                Parallel.For(0, tileCount*tileCount, t =>
                {
                    for (int y = 0; y < tileHeight; y++)
                        for (int x = 0; x < tileWidth; x++)
                        {
                            //We only have to translate the X and Y to the right spot on the screen...
                            //We tried implementing a morton curve, but we couldn't make a morton algorithm
                            //which was actually faster than this. We could possibly implement a lookup table,
                            //filled on initialization, but since this code works great, that's not a priority.
                            int screenY = y + (t/tileCount) * tileHeight;
                            int screenX = x + (t%tileCount) * tileWidth;
                            
                            // generate primary ray
                            Ray ray = camera.Generate(RTTools.GetRNG(), screenX, screenY);
                            // trace path
                            int pixelIdx = screenX + screenY * screen.width;
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