package core

type Adder3 struct {
	sum       Chan3
	terms     []RChan3
	blocksize int
}

func NewAdder3(sum Chan3, terms ...RChan3) *Adder3 {
	Assert(len(terms) > 1)
	for _, t := range terms {
		Assert(t.Size() == sum.Size())
	}
	return &Adder3{sum, terms, BlockLen(sum.Size())}
}

func (a *Adder3) Run() {
	in := make([][3][]float32, len(a.terms))
	for {
		// lock
		for i, t := range a.terms {
			in[i] = t.ReadNext(a.blocksize)
		}
		out := a.sum.WriteNext(a.blocksize)

		// add
		for c := 0; c < 3; c++ {
			for i := 0; i < len(out[0]); i++ {
				sum := in[0][c][i] + in[1][c][i]
				for j := 2; j < len(a.terms); j++ {
					sum += in[j][c][i]
				}
				out[c][i] = sum
			}
		}

		// unlock
		a.sum.WriteDone()
		for _, t := range a.terms {
			t.ReadDone()
		}
	}
}
