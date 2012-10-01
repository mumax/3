package mag

import (
	"nimble-cube/core"
)

type Stencil2D struct {
	in        core.RChan3
	out       core.Chan3
	W00       float32 // weight of central (0, 0) pixel
	W_10, W10 float32 // weight of top (-1,0), bottom (1,0) neighbor
	W0_1, W01 float32 // weight of left (0, -1), right (0, 1) neighbor
}

// New 2D 5-point stencil. Weights have to be set afterwards.
func NewStencil2D(in core.RChan3, out core.Chan3) *Stencil2D {
	core.Assert(in.Size()[0] == 1)
	core.Assert(in.Size() == out.Size())
	return &Stencil2D{in: in, out: out}
}

func (s *Stencil2D) Run() {
	core.Debug("running 4-neighbor 2D stencil")
	size := s.in.Size()
	bs := core.BlockSize(size)             // block size
	bl := core.Prod(bs)                    // elements per block
	bs2 := [3]int{bs[0], 2 * bs[1], bs[2]} // size of 2 blocks
	//bs1 := [3]int{bs[0], bs[1] + 1, bs[2]} // size of block + 1 line
	N := core.Prod(size) // elements per frame
	N1 := size[1]        // lines per frame
	//N2 := size[2]                          // lines per frame
	B1 := bs[1]      // lines per block
	nB := div(N, bl) // number of blocks

	for {
		in := s.in.ReadNext(bl)
		In := core.Reshape3(in, bs)

		out := s.out.WriteNext(bl)
		Out := core.Reshape3(out, bs)

		// very first line of entire frame
		s.span3(In, Out, 0, N1)

		for b := 0; b < nB-1; b++ {
			// block bulk
			for j := 1; j < B1-1; j++ {
				s.span3(In, Out, j, N1)
			}
			// block last line
			in = s.in.ReadDelta(0, bl)
			In := core.Reshape3(in, bs2)
			s.span3(In, Out, B1-1, N1)

			// next block first line
			out = s.out.WriteDelta(0, bl)
			Out = core.Reshape3(out, bs2)
			s.span3(In, Out, B1, N1)
			out = s.out.WriteDelta(bl, 0)
			Out = core.Reshape3(out, bs)
			in = s.in.ReadDelta(bl, 0)
			In = core.Reshape3(in, bs)
		}

		// last block bulk+last line
		for j := 1; j < B1; j++ {
			s.span3(In, Out, j, N1)
		}
		s.out.WriteDone()
		s.in.ReadDone()
	}
}

func (s *Stencil2D) span3(input, output [3][][][]float32, idx, N1 int) {
	for c := range input {
		s.span(input[c][0], output[c][0], idx, N1) // [0]: 2D
	}
}

// calculate line idx of stencil with total Y size N1.
// input/output may be small blocks of total arrays of size N1 x len(input[0])
// input is clamped for out-of-bounds elements.
func (s *Stencil2D) span(input, output [][]float32, idx, N1 int) {
	N2 := len(input[0])
	// line above idx, on indx and below idx
	// clamp out-of-bound lines
	in_1, in0, in1 := input[idx], input[idx], input[idx]
	if idx > 0 {
		in_1 = input[idx-1]
	}
	if idx < N1-1 {
		in1 = input[idx+1]
	}
	out := output[idx]

	w00 := s.W00
	w_10, w10 := s.W_10, s.W10
	w0_1, w01 := s.W0_1, s.W01

	{ // leftmost point
		k := 0
		out[k] = w_10*in_1[k] + w10*in1[k] + w0_1*in0[k-0] + w00*in0[k] + w01*in0[k+1]
		//                                        ^^^^^^^^
	} //                                          clamped

	// inner part of line
	for k := 1; k < N2-1; k++ {
		out[k] = w_10*in_1[k] + w10*in1[k] + w0_1*in0[k-1] + w00*in0[k] + w01*in0[k+1]
	}

	{ // rightmost point
		k := N2 - 1
		out[k] = w_10*in_1[k] + w10*in1[k] + w0_1*in0[k-1] + w00*in0[k] + w01*in0[k+0]
		//                                                                    ^^^^^^^^
	} //                                                                      clamped
}

func div(a, b int) int {
	if a%b != 0 {
		core.Panic(a, "%", b, "!=", 0)
	}
	return a / b
}
