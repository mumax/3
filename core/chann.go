package core

type ChanN []Chan1

func MakeChanN(nComp int, tag, unit string, m *Mesh, blocks ...int) ChanN {
	//tag = UniqueTag(tag)
	c := make(ChanN, nComp)
	for i := range c {
		c[i] = MakeChan(tag, unit, m, blocks...)
	}
	AddQuant(tag)
	return c
}

// WriteNext locks and returns a slice of length n for 
// writing the next n elements to the Chan3.
// When done, WriteDone() should be called to "send" the
// slice down the Chan3. After that, the slice is not valid any more.
func (c ChanN) WriteNext(n int) [3][]float32 {
	var next [3][]float32
	for i := range c {
		c[i].WriteNext(n)
		a, b := c[i].mutex.WRange()
		next[i] = c[i].slice.Slice(a, b).list
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

func (c ChanN) WriteDelta(Δstart, Δstop int) [3][]float32 {
	var next [3][]float32
	for i := range c {
		c[i].WriteDelta(Δstart, Δstop)
		a, b := c[i].mutex.WRange()
		next[i] = c[i].slice.Slice(a, b).list
	}
	return next
}

func (c ChanN) Mesh() *Mesh  { return c[0].Mesh }
func (c ChanN) Unit() string { return c[0].Unit() }
func (c ChanN) Tag() string  { return c[0].Tag() }
func (c ChanN) NComp() int   { return len(c) }

// UnsafeData returns the underlying storage without locking.
// Intended only for page-locking, not for reading or writing.
func (c ChanN) UnsafeData() [3][]float32 {
	return [3][]float32{c[0].slice.list, c[1].slice.list, c[2].slice.list}
}
func (c ChanN) UnsafeArray() [3][][][]float32 {
	return [3][][][]float32{c[0].slice.array, c[1].slice.array, c[2].slice.array}
}
func (c ChanN) Comp(idx int) Chan1 { return c[idx] }

func (c ChanN) Chan3() Chan3 {
	Assert(c.NComp() == 3)
	return Chan3{c[0], c[1], c[2]}
}

func (c ChanN) Chan1() Chan1 {
	Assert(c.NComp() == 1)
	return c[0]
}
