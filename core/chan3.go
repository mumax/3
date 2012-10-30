package core

// Chan of 3-vector data.
type Chan3 ChanN

func MakeChan3(tag, unit string, m *Mesh, blocks ...int) Chan3 {
	c := make(Chan3, 3)
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
func (c Chan3) WriteNext(n int) [3][]float32 {
	next := ChanN(c).WriteNext(n)
	return [3][]float32{next[0], next[1], next[2]}
	//var next [3][]float32
	//for i := range c {
	//	c[i].WriteNext(n)
	//	a, b := c[i].mutex.WRange()
	//	next[i] = c[i].slice.Slice(a, b).list
	//}
	//return next
}

// WriteDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan3.
func (c Chan3) WriteDone() {
	ChanN(c).WriteDone()
	//for i := range c {
	//	c[i].WriteDone()
	//}
}

func (c Chan3) WriteDelta(Δstart, Δstop int) [3][]float32 {
	next := ChanN(c).WriteDelta(Δstart, Δstop)
	return [3][]float32{next[0], next[1], next[2]}
	//var next [3][]float32
	//for i := range c {
	//	c[i].WriteDelta(Δstart, Δstop)
	//	a, b := c[i].mutex.WRange()
	//	next[i] = c[i].slice.Slice(a, b).list
	//}
	//return next
}

func (c Chan3) Mesh() *Mesh  { return ChanN(c).Mesh() }
func (c Chan3) Size() [3]int { return ChanN(c).Size() }
func (c Chan3) Unit() string { return ChanN(c).Unit() }
func (c Chan3) Tag() string  { return ChanN(c).Tag() }

// UnsafeData returns the underlying storage without locking.
// Intended only for page-locking, not for reading or writing.
func (c Chan3) UnsafeData() [3][]float32 {
	return [3][]float32{c[0].slice.list, c[1].slice.list, c[2].slice.list}
}

func (c Chan3) UnsafeArray() [3][][][]float32 {
	return [3][][][]float32{c[0].slice.array, c[1].slice.array, c[2].slice.array}
}
func (c Chan3) Comp(idx int) Chan1 { return c[idx] }

func (c Chan3) ChanN() ChanN { return ChanN(c) }
