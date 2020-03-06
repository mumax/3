// Applies the fft shift on raw complex fft data, and puts 
// the real and imaginary parts in separate arrays
extern "C" __global__ void
fftshift(  float * __restrict__ real, 
           float * __restrict__ imag,
           float * __restrict__ rawfft,
           int Nx, int Ny, int Nz) {

    // Cell index for the lhs (real and imag array)
    int3 i;
    i.x = blockIdx.x * blockDim.x + threadIdx.x;
    i.y = blockIdx.y * blockDim.y + threadIdx.y;
    i.z = blockIdx.z * blockDim.z + threadIdx.z;

    if(i.x>=Nx || i.y>=Ny || i.z>=Nz) {
        return;
    }

    // Unraveled cell index for the lhs
    int I = i.x + i.y*Nx + i.z*Nx*Ny;

    // Cell index for the rhs (the input array)
    int3 k;
    k.x = (i.x+(Nx+1)/2) % Nx; // apply fft shift
    k.y = (i.y+(Ny+1)/2) % Ny;
    k.z = (i.z+(Nz+1)/2) % Nz;

    // In the x direction, only the first Nx/2+1 of the values 
    // are stored in the raw fft data
    int Nx_ = Nx/2+1;
    
    // The other half of the values are simply the complex 
    // conjugates of the first half in reversed order
    int imagsgn = 1;
    if (k.x >= Nx_) {
        imagsgn = -1; // complex conjugate
        k.x = Nx-k.x; // reversed order
    }

    // Unraveled cell index of the rhs (the rawfft array)
    int K = k.x + k.y*Nx_ + k.z*Nx_*Ny;

    // Get the real and imaginary values
    real[I] = rawfft[2*K];
    imag[I] = imagsgn * rawfft[2*K+1];
}
