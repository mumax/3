// dst[i] = pow(a[i], b[i]), pow(a,b) for negative a and b returns -pow(-a,b) for fractional exponents, and for 0^0 returns 1
extern "C" __global__ void
pw_pow(float* __restrict__ dst, float* __restrict__ a, float* __restrict__ b, int N) {
int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        float x = a[i];
        float y = b[i];

        // Guard: 0^negative → return 0
        if (x == 0.0f && y < 0.0f) {
            dst[i] = 0.0f;
            return;
        }

        // Check if exponent is effectively integer
        float y_floor = floorf(y);
        bool y_is_int = fabsf(y - y_floor) < 1e-6f;

        if (x < 0.0f && !y_is_int) {
            // Fractional exponent → move minus sign outside
            dst[i] = -1.0f * powf(fabsf(x), y);
        } else {
            // Integer exponent or positive base
            dst[i] = powf(x, y);
        }
    }
}