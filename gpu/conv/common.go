package conv

// common data for all convolutions
type common struct {
	size          [3]int       // 3D size of the input/output data
	kernSize      [3]int       // Size of kernel and logical FFT size.
	n             int          // product of size
	input, output [3][]float32 // input/output arrays, 3 component vectors
}
