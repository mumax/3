package cuda

import (
	"github.com/barnex/cuda5/cu"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
	"github.com/mumax/3/util"
	"log"
)

// Stores the necessary state to perform FFT-accelerated convolution
// with magnetostatic kernel (or other kernel of same symmetry).
type DemagConvolution struct {
	size        [3]int            // 3D size of the input/output data
	kernSize    [3]int            // Size of kernel and logical FFT size.
	fftKernSize [3]int            // Size of real, FFTed kernel
	fftRBuf     [3]*data.Slice    // FFT input buf for FFT, shares storage with fftCBuf.
	fftCBuf     [3]*data.Slice    // FFT output buf, shares storage with fftRBuf
	gpuFFTKern  [3][3]*data.Slice // FFT kernel on device
	fwPlan      fft3DR2CPlan      // Forward FFT (1 component)
	bwPlan      fft3DC2RPlan      // Backward FFT (1 component)
	kern        [3][3]*data.Slice // Real-space kernel (host), removed after self-test
	FFTMesh     data.Mesh         // mesh of FFT m
}

func (c *DemagConvolution) init() {
	{ // init FFT plans
		padded := c.kernSize
		c.fwPlan = newFFT3DR2C(padded[X], padded[Y], padded[Z], stream[0])
		c.bwPlan = newFFT3DC2R(padded[X], padded[Y], padded[Z], stream[0])
	}

	{ // init device buffers
		// 2D re-uses fftBuf[X] as fftBuf[Z], 3D needs all 3 fftBufs.
		nc := fftR2COutputSizeFloats(c.kernSize)
		c.fftCBuf[X] = makeFloats(nc)
		c.fftCBuf[Y] = makeFloats(nc)
		if c.is3D() {
			c.fftCBuf[Z] = makeFloats(nc)
		} else {
			c.fftCBuf[Z] = c.fftCBuf[X]
		}
		for i := 0; i < 3; i++ {
			c.fftRBuf[i] = c.fftCBuf[i].Slice(0, prod(c.kernSize))
		}
	}

	if c.is2D() {
		c.initFFTKern2D()
	} else {
		c.initFFTKern3D()
	}
}

// initialize FFT(Kernel) for 3D
func (c *DemagConvolution) initFFTKern3D() {

	// size of FFT(kernel): store real parts only
	c.fftKernSize = fftR2COutputSizeFloats(c.kernSize)
	util.Assert(c.fftKernSize[X]%2 == 0)
	c.fftKernSize[X] /= 2

	// will store only 1/4 (symmetry), but not yet
	halfkern := c.fftKernSize

	output := c.fftCBuf[0]
	input := c.fftRBuf[0]

	fftKern := data.NewSlice(1, data.NewMesh(halfkern[X], halfkern[Y], halfkern[Z], 1, 1, 1)) // host
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ { // upper triangular part
			if c.kern[i][j] != nil { // ignore 0's
				data.Copy(input, c.kern[i][j])
				c.fwPlan.Exec(input, output)
				scaleRealParts(fftKern, output.Slice(0, prod(halfkern)*2), 1/float32(c.fwPlan.InputLen()))
				c.gpuFFTKern[i][j] = GPUCopy(fftKern)
			}
		}
	}
}

// Initialize GPU FFT kernel for 2D.
// Only the non-redundant parts are stored on the GPU.
func (c *DemagConvolution) initFFTKern2D() {

	// size of FFT(kernel): store real parts only
	c.fftKernSize = fftR2COutputSizeFloats(c.kernSize)
	util.Assert(c.fftKernSize[X]%2 == 0)
	c.fftKernSize[X] /= 2

	// store only 1/2 (symmetry), not yet
	halfkern := c.fftKernSize
	//halfkern[Y] = halfkern[Y]/2 + 1

	output := c.fftCBuf[0]
	input := c.fftRBuf[0]

	// upper triangular part
	fftKern := data.NewSlice(1, data.NewMesh(halfkern[X], halfkern[Y], halfkern[Z], 1, 1, 1)) // host
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ { // upper triangular part
			if c.kern[i][j] != nil { // ignore 0's
				data.Copy(input, c.kern[i][j])
				c.fwPlan.Exec(input, output)
				scaleRealParts(fftKern, output.Slice(0, prod(halfkern)*2), 1/float32(c.fwPlan.InputLen()))
				c.gpuFFTKern[i][j] = GPUCopy(fftKern)
			}
		}
	}
}

// Calculate the demag field of m * vol * Bsat, store result in B.
// 	m:    magnetization normalized to unit length
// 	vol:  unitless mask used to scale m's length, may be nil
// 	Bsat: saturation magnetization in Tesla
// 	B:    resulting demag field, in Tesla
func (c *DemagConvolution) Exec(B, m, vol *data.Slice, Bsat LUTPtr, regions *Bytes) {
	if c.is2D() {
		c.exec2D(B, m, vol, Bsat, regions)
	} else {
		c.exec3D(B, m, vol, Bsat, regions)
	}
}

// zero 1-component slice
func zero1_async(dst *data.Slice, str int) {
	cu.MemsetD32Async(cu.DevicePtr(uintptr(dst.DevPtr(0))), 0, int64(dst.Len()), stream[str])
}

// forward FFT component i
func (c *DemagConvolution) fwFFT(i int, inp, vol *data.Slice, Bsat LUTPtr, regions *Bytes) {
	zero1_async(c.fftRBuf[i], stream0)
	in := inp.Comp(i)
	copyPadMul(c.fftRBuf[i], in, vol, c.kernSize, c.size, Bsat, regions, stream0)
	c.fwPlan.ExecAsync(c.fftRBuf[i], c.fftCBuf[i])
}

// backward FFT component i
func (c *DemagConvolution) bwFFT(i int, outp *data.Slice) {
	c.bwPlan.ExecAsync(c.fftCBuf[i], c.fftRBuf[i])
	out := outp.Comp(i)
	copyUnPad(out, c.fftRBuf[i], c.size, c.kernSize, stream0)
}

// forward FFT of magnetization one component.
// returned slice is valid until next FFT or convolution
func (c *DemagConvolution) FFT(m, vol *data.Slice, comp int, Bsat LUTPtr, regions *Bytes) *data.Slice {
	c.fwFFT(comp, m, vol, Bsat, regions)
	return c.fftCBuf[comp]
}

func (c *DemagConvolution) exec3D(outp, inp, vol *data.Slice, Bsat LUTPtr, regions *Bytes) {
	for i := 0; i < 3; i++ { // FW FFT
		c.fwFFT(i, inp, vol, Bsat, regions)
	}

	// kern mul
	Nx, Ny, Nz := c.fftKernSize[X], c.fftKernSize[Y], c.fftKernSize[Z]
	kernMulRSymm3D_async(c.fftCBuf,
		c.gpuFFTKern[X][X], c.gpuFFTKern[Y][Y], c.gpuFFTKern[Z][Z],
		c.gpuFFTKern[Y][Z], c.gpuFFTKern[X][Z], c.gpuFFTKern[X][Y],
		Nx, Ny, Nz, stream0)

	for i := 0; i < 3; i++ { // BW FFT
		c.bwFFT(i, outp)
	}
	//SyncAll()
}

func (c *DemagConvolution) exec2D(outp, inp, vol *data.Slice, Bsat LUTPtr, regions *Bytes) {
	// Convolution is separated into
	// a 1D convolution for x and a 2D convolution for yz.
	// So only 2 FFT buffers are needed at the same time.

	dbg("exec2D")
	dbg("inp:", inp.HostCopy())

	c.fwFFT(Z, inp, vol, Bsat, regions) // FFT Z

	dbg("fftmz:", c.fftCBuf[Z].HostCopy())

	// kern mul Z
	Nx, Ny := c.fftKernSize[X], c.fftKernSize[Y]
	kernMulRSymm2Dz_async(c.fftCBuf[Z], c.gpuFFTKern[Z][Z], Nx, Ny, stream0)

	dbg("fftBz:", c.fftCBuf[Z].HostCopy())

	c.bwFFT(Z, outp) // bw FFT z

	dbg("Bz:", outp.Comp(Z).HostCopy())

	// FW FFT xy
	c.fwFFT(X, inp, vol, Bsat, regions)
	c.fwFFT(Y, inp, vol, Bsat, regions)
	dbg("fftmx:", c.fftCBuf[X].HostCopy())
	dbg("fftmy:", c.fftCBuf[Y].HostCopy())

	// kern mul xy
	kernMulRSymm2Dxy_async(c.fftCBuf[X], c.fftCBuf[Y],
		c.gpuFFTKern[X][X], c.gpuFFTKern[Y][Y], c.gpuFFTKern[X][Y],
		Nx, Ny, stream0)

	dbg("fftBx:", c.fftCBuf[X].HostCopy())
	dbg("fftBy:", c.fftCBuf[Y].HostCopy())

	// BW FFT xy
	c.bwFFT(X, outp)
	c.bwFFT(Y, outp)

	dbg("Bx:", outp.Comp(X).HostCopy())
	dbg("By:", outp.Comp(Y).HostCopy())
	//SyncAll()
}

func (c *DemagConvolution) is2D() bool {
	return false // !!!!!!!!!!!!!!!!!!!!!!!
	return c.size[Z] == 1
}

func (c *DemagConvolution) is3D() bool {
	return !c.is2D()
}

// Initializes a demag convolution for the given mesh geometry and magnetostatic kernel.
func newConvolution(mesh *data.Mesh, kernel [3][3]*data.Slice) *DemagConvolution {
	size := mesh.Size()
	c := new(DemagConvolution)
	c.size = size
	c.kern = kernel
	c.kernSize = kernel[X][X].Mesh().Size()
	c.init()
	c.initFFTMesh()

	//dbg("kernel", c.kern)

	//for i, k := range c.gpuFFTKern {
	//	for j, k := range k {
	//		if k != nil {
	//			dbg("fftkernel", i, j, k.HostCopy())
	//		}
	//	}
	//}

	testConvolution(c, mesh)
	c.freeKern()
	return c
}

// release the real-space kernel so the host memory can be reclaimed by GC.
func (c *DemagConvolution) freeKern() {
	for i := range c.kern {
		for j := range c.kern[i] {
			c.kern[i][j] = nil
		}
	}
}

// Mesh for FFT(m) quantity, etc.
func (c *DemagConvolution) initFFTMesh() {
	n := fftR2COutputSizeFloats(c.kernSize)
	cell := c.kern[0][0].Mesh().CellSize()
	c.FFTMesh = *data.NewMesh(n[X], n[Y], n[Z], 1/cell[X], 1/cell[Y], 1/cell[Z])
	c.FFTMesh.Unit = "/m"
}

// Default accuracy setting for demag kernel.
const DEFAULT_KERNEL_ACC = 6

// Initializes a convolution to evaluate the demag field for the given mesh geometry.
func NewDemag(mesh *data.Mesh) *DemagConvolution {
	k := mag.BruteKernel(mesh, DEFAULT_KERNEL_ACC)
	return newConvolution(mesh, k)
}

func dbg(msg ...interface{}) {
	for _, m := range msg {
		log.Println(m)
	}
}
