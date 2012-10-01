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
	return &Stencil2D{in: in, out: out}
}

func (s *Stencil2D) Run() {
	core.Debug("running 5-point stencil")
	size := s.in.Size()
	N := core.Prod(size)
	//	bl := core.BlockLen(s.in.Size())
	//	nB := div(N , bl)
	//	bs := core.BlockSize(s.in.Size())
	//	bs2 := bs // size of two subsequent m blocks
	//	bs[1] *= 2 

	for {
		in := s.in.ReadNext(N)
		In := core.Reshape3(in, size) // todo: bs
		out := s.out.WriteNext(N)
		Out := core.Reshape3(out, size)

		for c := range In {
			// i=0 (2D)
			for j := 0; j < len(In[c][0]); j++ {
				s.span(In[c][0], Out[c][0], j)
			}
		}

		s.out.WriteDone()
		s.in.ReadDone()
	}
}

// calculate 1 line of the stencil.
func (s *Stencil2D) span(input, output [][]float32, idx int) {
	N1 := len(input)
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
