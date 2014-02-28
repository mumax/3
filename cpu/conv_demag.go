package cpu

import (
	"github.com/barnex/fftw"
	"github.com/mumax/3/data"
)

// Stores the necessary state to perform FFT-accelerated convolution
// with magnetostatic kernel (or other kernel of same symmetry).
type DemagConvolution struct {
	inputSize    [3]int            // 3D size of the input/output data
	realKernSize [3]int            // Size of kernel and logical FFT size.
	fftKernSize  [3]int            // Size FFTed kernel, real parts only
	fftRBuf      [3]*data.Slice    // FFT input buf
	fftCBuf      [3]*data.Slice    // FFT output buf
	kern         [3][3]*data.Slice // FFT kernel on device
	fwPlan       [3]*fftw.R2CPlan  // Forward FFT
	bwPlan       [3]*fftw.C2RPlan  // Backward FFT
}

// Initializes a convolution to evaluate the demag field for the given grid size.
func NewDemag(inputSize, PBC [3]int, kernel [3][3]*data.Slice) *DemagConvolution {
	c := new(DemagConvolution)
	c.inputSize = inputSize
	c.realKernSize = kernel[X][X].Size()
	c.init(kernel)
	//testConvolution(c, PBC, kernel)
	return c
}

//// Calculate the demag field of m * vol * Bsat, store result in B.
//// 	m:    magnetization normalized to unit length
//// 	vol:  unitless mask used to scale m's length, may be nil
//// 	Bsat: saturation magnetization in Tesla
//// 	B:    resulting demag field, in Tesla
//func (c *DemagConvolution) Exec(B, m, vol *data.Slice, Bsat LUTPtr, regions *Bytes) {
//	if c.is2D() {
//		c.exec2D(B, m, vol, Bsat, regions)
//	} else {
//		c.exec3D(B, m, vol, Bsat, regions)
//	}
//}
//
//func (c *DemagConvolution) exec3D(outp, inp, vol *data.Slice, Bsat LUTPtr, regions *Bytes) {
//	for i := 0; i < 3; i++ { // FW FFT
//		c.fwFFT(i, inp, vol, Bsat, regions)
//	}
//
//	// kern mul
//	kernMulRSymm3D_async(c.fftCBuf,
//		c.kern[X][X], c.kern[Y][Y], c.kern[Z][Z],
//		c.kern[Y][Z], c.kern[X][Z], c.kern[X][Y],
//		c.fftKernSize[X], c.fftKernSize[Y], c.fftKernSize[Z])
//
//	for i := 0; i < 3; i++ { // BW FFT
//		c.bwFFT(i, outp)
//	}
//}

//func (c *DemagConvolution) exec2D(outp, inp, vol *data.Slice, Bsat LUTPtr, regions *Bytes) {
//
//	for i:=0; i<3; i++{
//		c.fwPlan[i].Execute()
//	}
//
//	//zero1_async(c.fftRBuf[i])
//	//in := inp.Comp(i)
//	//copyPadMul(c.fftRBuf[i], in, vol, c.realKernSize, c.inputSize, Bsat, regions)
//	//c.fwPlan.ExecAsync(c.fftRBuf[i], c.fftCBuf[i])
//
//	//// Convolution is separated into
//	//// a 1D convolution for z and a 2D convolution for xy.
//	//// So only 2 FFT buffers are needed at the same time.
//	//Nx, Ny := c.fftKernSize[X], c.fftKernSize[Y]
//
//	//// Z
//	//c.fwFFT(Z, inp, vol, Bsat, regions)
//	//kernMulRSymm2Dz_async(c.fftCBuf[Z], c.kern[Z][Z], Nx, Ny)
//	//c.bwFFT(Z, outp)
//
//	//// XY
//	//c.fwFFT(X, inp, vol, Bsat, regions)
//	//c.fwFFT(Y, inp, vol, Bsat, regions)
//	//kernMulRSymm2Dxy_async(c.fftCBuf[X], c.fftCBuf[Y],
//	//	c.kern[X][X], c.kern[Y][Y], c.kern[X][Y], Nx, Ny)
//	//c.bwFFT(X, outp)
//	//c.bwFFT(Y, outp)
//}

//func (c *DemagConvolution) is2D() bool {
//	return c.inputSize[Z] == 1
//}
//
//// zero 1-component slice
//func zero1_async(dst *data.Slice) {
//	cu.MemsetD32Async(cu.DevicePtr(uintptr(dst.DevPtr(0))), 0, int64(dst.Len()), stream0)
//}
//
//// forward FFT component i
//func (c *DemagConvolution) fwFFT(i int, inp, vol *data.Slice, Bsat LUTPtr, regions *Bytes) {
//	zero1_async(c.fftRBuf[i])
//	in := inp.Comp(i)
//	copyPadMul(c.fftRBuf[i], in, vol, c.realKernSize, c.inputSize, Bsat, regions)
//	c.fwPlan.ExecAsync(c.fftRBuf[i], c.fftCBuf[i])
//}
//
//// backward FFT component i
//func (c *DemagConvolution) bwFFT(i int, outp *data.Slice) {
//	c.bwPlan.ExecAsync(c.fftCBuf[i], c.fftRBuf[i])
//	out := outp.Comp(i)
//	copyUnPad(out, c.fftRBuf[i], c.inputSize, c.realKernSize)
//}

var fftFlags = fftw.MEASURE

func (c *DemagConvolution) init(realKern [3][3]*data.Slice) {

	// FFT plan sizes (z, y, x)
	logicSize := []int{c.realKernSize[Z], c.realKernSize[Y], c.realKernSize[X]}
	inembed := logicSize
	onembed := []int{logicSize[0], logicSize[1], (logicSize[2]/2 + 1)}
	onfloat := []int{onembed[0], onembed[1], onembed[2] * 2}

	// FFT buffers
	for i := 0; i < 3; i++ {
		c.fftRBuf[i] = data.NewSlice(1, [3]int{inembed[2], inembed[1], inembed[0]})
		c.fftCBuf[i] = data.NewSlice(1, [3]int{onfloat[2], onfloat[1], onfloat[0]})
	}

	// FFT plans
	stride := 1
	dist := 0
	for i := 0; i < 3; i++ {
		in := c.fftRBuf[i].Host()[0]
		out := fftw.CastRtoC(c.fftCBuf[i].Host()[0])
		c.fwPlan[i] = fftw.PlanManyR2C(logicSize, 1, in, inembed, stride, dist, out, onembed, stride, dist, fftFlags)
		c.bwPlan[i] = fftw.PlanManyC2R(logicSize, 1, out, onembed, stride, dist, in, inembed, stride, dist, fftFlags)
	}

	//	// init FFT kernel
	//	c.fftKernSize = fftR2COutputSizeFloats(c.realKernSize)
	//	// size of FFT(kernel): store real parts only
	//	util.Assert(c.fftKernSize[X]%2 == 0)
	//	c.fftKernSize[X] /= 2
	//
	//	//halfkern := [3]int{c.fftKernSize[X], c.fftKernSize[Y]/2 + 1, c.fftKernSize[Z]/2 + 1}
	//	halfkern := [3]int{c.fftKernSize[X], c.fftKernSize[Y], c.fftKernSize[Z]}
	//
	//	output := c.fftCBuf[0]
	//	input := c.fftRBuf[0]
	//
	//	fftKern := data.NewSlice(1, halfkern) // host
	//	for i := 0; i < 3; i++ {
	//		for j := i; j < 3; j++ { // upper triangular part
	//			if realKern[i][j] != nil { // ignore 0's
	//				data.Copy(input, realKern[i][j])
	//				//util.Println("input")
	//				//util.Printf("% 6f", input.HostCopy().Scalars())
	//				c.fwPlan.ExecAsync(input, output)
	//				//util.Println("output")
	//				//util.Printf("% 6f", output.HostCopy().Scalars())
	//				scaleRealParts(fftKern, output, 1/float32(c.fwPlan.InputLen()))
	//				c.kern[i][j] = GPUCopy(fftKern)
	//				//util.Println("fftK", i, j)
	//				//util.Printf("% 7f", fftKern.Scalars())
	//				//util.Println()
	//			}
	//		}
	//	}
}

//func (c *DemagConvolution) Free() {
//	if c == nil {
//		return
//	}
//	c.inputSize = [3]int{}
//	c.realKernSize = [3]int{}
//	for i := 0; i < 3; i++ {
//		c.fftCBuf[i].Free() // shared with fftRbuf
//		c.fftCBuf[i] = nil
//		c.fftRBuf[i] = nil
//
//		for j := 0; j < 3; j++ {
//			c.kern[i][j].Free()
//			c.kern[i][j] = nil
//		}
//		c.fwPlan.Free()
//		c.bwPlan.Free()
//	}
//}

const (
	X = 0
	Y = 1
	Z = 2
)
