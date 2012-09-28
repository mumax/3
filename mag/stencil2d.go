package mag

import (
	"nimble-cube/core"
)

type Stencil2D struct {
	in  core.RChan3
	out core.Chan3
	//weights
}

func (s *Stencil2D) Run() {
	size := s.in.Size()
	N := core.Prod(size)
	//	bl := core.BlockLen(s.in.Size())
	//	nB := div(N , bl)
	//	bs := core.BlockSize(s.in.Size())
	//	bs2 := bs // size of two subsequent m blocks
	//	bs[1] *= 2 

	for {
		in := s.in.ReadNext(N)
		In := core.Reshape3(in, size) // 2D // todo: bs
		out := s.out.WriteNext(N)
		Out := core.Reshape3(out, size)

		for c := range In {
			for i := 0; i < len(In[c]); i++ {
				s.span(In[c][0], Out[c][0], i)
			}
		}

		s.out.WriteDone()
		s.in.ReadDone()
	}
}

// calculate 1 line of the stencil.
func (s *Stencil2D) span(input, output [][]float32, idx int) {
	n := len(input)
	// line above idx, on indx and below idx
	// clamp out-of-bound lines
	in0, in1, in2 := input[idx], input[idx], input[idx]
	if idx > 0 {
		in0 = input[idx-1]
	}
	if idx < n-1 {
		in0 = input[idx+1]
	}
	out := output[idx]

	// leftmost point: clamp
	{
		i := 0
		out[i] = in1[i] + in2[i] + in0[i-0] + in0[i] + in0[i+1]
	}

	// inner part of line
	for i := 1; i < n-1; i++ {
		out[i] = in1[i] + in2[i] + in0[i-1] + in0[i] + in0[i+1]
	}

	// rightmost point: clamp
	{
		i := n - 1
		out[i] = in1[i] + in2[i] + in0[i-1] + in0[i] + in0[i+0]
	}
}

func NewStencil2D(in core.RChan3, out core.Chan3) *Stencil2D {
	return &Stencil2D{in, out}
}

// Naive implementation of 6-neighbor exchange field.
// Aex in Tm² (exchange stiffness divided by Msat0).
// Hex in Tesla.
//func exchange2d(m [3][][][]float32, Hex [3][][][]float32, cellsize [3]float64, aex_reduced float64) {
//	var (
//		facI = float32(aex_reduced / (cellsize[0] * cellsize[0]))
//		facJ = float32(aex_reduced / (cellsize[1] * cellsize[1]))
//		facK = float32(aex_reduced / (cellsize[2] * cellsize[2]))
//	)
//	N0, N1, N2 := len(m[0]), len(m[0][0]), len(m[0][0][0])
//
//}

func div(a, b int) int {
	if a%b != 0 {
		core.Panic(a, "%", b, "!=", 0)
	}
	return a / b
}
