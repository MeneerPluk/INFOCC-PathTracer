using System;
using System.IO;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;

namespace Template
{
	public class OpenTKApp : GameWindow
	{
		static int screenID;
		static Game game;
		static bool useGPU = true, terminated = false;
		static int runningTime = -1, gpuPlatform = 0;
		protected override void OnLoad( EventArgs e )
		{
			// called upon app init
			GL.ClearColor( Color.Black );
			GL.Enable( EnableCap.Texture2D );
			GL.Hint( HintTarget.PerspectiveCorrectionHint, HintMode.Nicest );
			Width = 800;
			Height = 480;
			game = new Game();
			game.screen = new Surface( Width, Height );
			Sprite.target = game.screen;
			screenID = game.screen.GenTexture();
			game.Init( runningTime, useGPU, gpuPlatform );
		}
		protected override void OnUnload( EventArgs e )
		{
			// called upon app close
			GL.DeleteTextures( 1, ref screenID );
			Environment.Exit( 0 ); // bypass wait for key on CTRL-F5
		}
		protected override void OnResize( EventArgs e )
		{
			// called upon window resize
			GL.Viewport(0, 0, Width, Height);
			GL.MatrixMode( MatrixMode.Projection );
			GL.LoadIdentity();
			GL.Ortho( -1.0, 1.0, -1.0, 1.0, 0.0, 4.0 );
		}
		protected override void OnUpdateFrame( FrameEventArgs e )
		{
			// called once per frame; app logic
			var keyboard = OpenTK.Input.Keyboard.GetState();
			if (keyboard[OpenTK.Input.Key.Escape]) this.Exit();
		}
		protected override void OnRenderFrame( FrameEventArgs e )
		{
			// called once per frame; render
			game.Tick();
			if (terminated) 
			{
				Exit();
				return;
			}
			GL.BindTexture( TextureTarget.Texture2D, screenID );
			GL.TexImage2D( TextureTarget.Texture2D, 
						   0, 
						   PixelInternalFormat.Rgba, 
						   game.screen.width, 
						   game.screen.height, 
						   0, 
						   OpenTK.Graphics.OpenGL.PixelFormat.Bgra, 
						   PixelType.UnsignedByte, 
						   game.screen.pixels 
						 );
			GL.Clear( ClearBufferMask.ColorBufferBit );
			GL.MatrixMode( MatrixMode.Modelview );
			GL.LoadIdentity();
			GL.BindTexture( TextureTarget.Texture2D, screenID );
			GL.Begin( PrimitiveType.Quads );
			GL.TexCoord2( 0.0f, 1.0f ); GL.Vertex2( -1.0f, -1.0f );
			GL.TexCoord2( 1.0f, 1.0f ); GL.Vertex2(  1.0f, -1.0f );
			GL.TexCoord2( 1.0f, 0.0f ); GL.Vertex2(  1.0f,  1.0f );
			GL.TexCoord2( 0.0f, 0.0f ); GL.Vertex2( -1.0f,  1.0f );
			GL.End();
			SwapBuffers();
		}
		[STAThread]
		public static void Main( string[] args ) 
		{ 
			// parse command line parameters
			if (args.Length == 3)
			{
				if (args[0] == "G") useGPU = true; else useGPU = false;
				gpuPlatform = Int32.Parse( args[1] );
				runningTime = Int32.Parse( args[2] );
			}
			// entry point
			using (OpenTKApp app = new OpenTKApp()) 
			{ 
				app.Run( 30.0, 0.0 ); 
			}
		}
		static public void Report( int ms, int spp, Surface screen )
		{
			// get system information
			int RealCores = 0, VirtualCores = System.Environment.ProcessorCount;
			try
			{
				foreach (var item in new System.Management.ManagementObjectSearcher( "Select NumberOfCores from Win32_Processor" ).Get())
					RealCores += int.Parse(item["NumberOfCores"].ToString());
			}
			catch { RealCores = System.Environment.ProcessorCount; }
			// calculate RMSE
			Surface reference = new Surface( "../../assets/reference.png" );
			float SE = 0;
			for( int y = 0; y < screen.height; y++ ) for( int x = 0; x < screen.width; x++ )
			{
				int p1 = reference.pixels[x + y * reference.width], p2 = screen.pixels[x + y * screen.width];
				int dr = ((p1 >> 16) & 255) - ((p2 >> 16) & 255);
				int dg = ((p1 >> 8) & 255) - ((p2 >> 8) & 255);
				int db = (p1 & 255) - (p2 & 255);
				SE += (float)(dr * dr + dg * dg + db * db);
			}
			float RMSE = (float)Math.Sqrt( SE / (float)(screen.width * screen.height) );
			// report
			Console.Write( spp.ToString() + " " +  
						   ms.ToString() + " " +
						   RMSE.ToString() + " " +
						   RealCores.ToString() + " " +
						   VirtualCores.ToString() ); 
			terminated = true;
		}
	}
}