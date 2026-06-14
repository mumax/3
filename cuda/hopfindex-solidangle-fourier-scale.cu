// Rescale the effective field to a unit system where the cell spacing = 1
extern "C" __global__ void
scaleemergentfield(float* __restrict__  Fx_scale,  float* __restrict__  Fy_scale,  float* __restrict__  Fz_scale,
                float* __restrict__  Fx,  float* __restrict__  Fy,  float* __restrict__  Fz,
                float cx, float cy, float cz, int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix>= Nx || iy>= Ny || iz>=Nz) {
        return;
    }

    int I = (iz*Ny + iy)*Nx + ix;

    Fx_scale[I] = Fx[I] * cy * cz;
    Fy_scale[I] = Fy[I] * cx * cz;
    Fz_scale[I] = Fz[I] * cx * cy;
}
