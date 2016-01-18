using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

// special file loading code to reduce time for loading
// the HDR skymap.

namespace Template {

unsafe public class WinFileIO : IDisposable
{
	// based on Robert G. Bryan
	private const uint GENERIC_READ = 0x80000000;
	private const uint OPEN_EXISTING = 3;

	private GCHandle gchBuf;            // handle to GCHandle object used to pin the I/O buffer in memory.
	private System.IntPtr pHandle;      // handle to the file to be read from or written to.
	private void* pBuffer;              // pointer to the buffer used to perform I/O.

	[System.Runtime.InteropServices.DllImport("kernel32", SetLastError = true)]
	static extern unsafe System.IntPtr CreateFile
	(
		string FileName,         
		uint DesiredAccess, 
		uint ShareMode, 
		uint SecurityAttributes, 
		uint CreationDisposition, 
		uint FlagsAndAttributes,
		int hTemplateFile        
	);
	[System.Runtime.InteropServices.DllImport("kernel32", SetLastError = true)]
	static extern unsafe bool ReadFile
	(
		System.IntPtr hFile,    
		void* pBuffer,          
		int NumberOfBytesToRead,
		int* pNumberOfBytesRead,
		int Overlapped          
	);
	[System.Runtime.InteropServices.DllImport("kernel32", SetLastError = true)]
	static extern unsafe bool CloseHandle( System.IntPtr hObject );
	public WinFileIO()
	{
		pHandle = IntPtr.Zero;
	}
	public WinFileIO(Array Buffer)
	{
		if (gchBuf.IsAllocated) gchBuf.Free();
		gchBuf = GCHandle.Alloc(Buffer, GCHandleType.Pinned);
		IntPtr pAddr = Marshal.UnsafeAddrOfPinnedArrayElement(Buffer, 0);
		pBuffer = (void*)pAddr.ToPointer();
		pHandle = IntPtr.Zero;
	}
	protected void Dispose(bool disposing)
	{
		Close();
		if (gchBuf.IsAllocated) gchBuf.Free();
	}
	public void Dispose()
	{
		Dispose(true);
		GC.SuppressFinalize(this);
	}
	~WinFileIO()
	{
		Dispose(false);
	}
	public void OpenForReading(string FileName)
	{
		Close();
		pHandle = CreateFile( FileName, GENERIC_READ, 0, 0, OPEN_EXISTING, 0, 0 );
		if (pHandle == System.IntPtr.Zero)
		{
			Win32Exception WE = new Win32Exception();
			ApplicationException AE = new ApplicationException("WinFileIO:OpenForReading - Could not open file " + FileName + " - " + WE.Message);
			throw AE;
		}
	}
	public int Read(int BytesToRead)
	{
		int BytesRead = 0;
		if (!ReadFile( pHandle, pBuffer, BytesToRead, &BytesRead, 0 ))
		{
			Win32Exception WE = new Win32Exception();
			ApplicationException AE = new ApplicationException( "WinFileIO:Read - Error occurred reading a file. - " + WE.Message);
			throw AE;
		}
		return BytesRead;
	}
	public bool Close()
	{
		bool Success = true;
		if (pHandle != IntPtr.Zero)
		{
			Success = CloseHandle( pHandle );
			pHandle = IntPtr.Zero;
		}
		return Success;
	}
}

} // namespace Template
