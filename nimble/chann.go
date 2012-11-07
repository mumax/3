package nimble

type ChanN []Chan1

func MakeChanN(nComp int, tag, unit string, m *Mesh, memType MemType, blocks ...int) ChanN {
	c := make(ChanN, nComp)
	for i := range c {
		c[i] = MakeChan(tag, unit, m, memType, blocks...)
	}
	AddQuant(tag)
	return c
}

func (c ChanN) Mesh() *Mesh  { return c[0].Mesh }
func (c ChanN) Size() [3]int { return c[0].Size() }
func (c ChanN) Unit() string { return c[0].Unit() }
func (c ChanN) Tag() string  { return c[0].Tag() }
func (c ChanN) NComp() int   { return len(c) }
func (c ChanN) Comp(i int) Chan1   { return c[i] }

func (c ChanN) Chan3() Chan3 {
	Assert(c.NComp() == 3)
	return Chan3{c[0], c[1], c[2]}
}

func (c ChanN) Chan1() Chan1 {
	Assert(c.NComp() == 1)
	return c[0]
}

// WriteNext locks and returns a slice of length n for 
// writing the next n elements to the Chan3.
// When done, WriteDone() should be called to "send" the
// slice down the Chan3. After that, the slice is not valid any more.
func (c ChanN) WriteNext(n int) []Slice {
	next := make([]Slice, c.NComp())
	for i := range c {
		c[i].WriteNext(n)
		a, b := c[i].mutex.WRange()
		next[i] = c[i].slice.Slice(a, b)
	}
	return next
}

// WriteDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan3.
func (c ChanN) WriteDone() {
	for i := range c {
		c[i].WriteDone()
	}
}

//func (c ChanN) WriteDelta(Δstart, Δstop int) [][]float32 {
//	next := make([][]float32, c.NComp())
//	for i := range c {
//		c[i].WriteDelta(Δstart, Δstop)
//		a, b := c[i].mutex.WRange()
//		next[i] = c[i].slice.Slice(a, b).list
//	}
//	return next
//}
