package core

// Read-only N-component Chan.
type RChanN []RChan1

func (c ChanN) NewReader() RChanN {
	return RChanN{c[0].NewReader(), c[1].NewReader(), c[2].NewReader()}
}

func (c RChanN) Mesh() *Mesh  { return c[0].Mesh }
func (c RChanN) Unit() string { return c[0].Unit() }
func (c RChanN) Tag() string  { return c[0].Tag() }
func (c RChanN) NComp() int   { return len(c) }

// ReadNext locks and returns a slice of length n for 
// reading the next n elements from the Chan.
// When done, ReadDone() should be called .
// After that, the slice is not valid any more.
func (c RChanN) ReadNext(n int) []Slice {
	next := make([]Slice, c.NComp())
	for i := range c {
		c[i].mutex.ReadNext(n)
		a, b := c[i].mutex.RRange()
		next[i] = c[i].slice.Slice(a, b)
	}
	return next
}

// temporary
func (c RChanN) ReadNextList(n int) [][]float32 {
	next := make([][]float32, c.NComp())
	for i := range c {
		c[i].mutex.ReadNext(n)
		a, b := c[i].mutex.RRange()
		next[i] = c[i].slice.Slice(a, b).Host()
	}
	return next
}

// ReadDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan3.
func (c RChanN) ReadDone() {
	for i := range c {
		c[i].ReadDone()
	}
}

// UnsafeData returns the underlying storage without locking.
// Intended only for page-locking, not for reading or writing.
//func (c RChanN) UnsafeData() [3][]float32 {
//	return [3][]float32{c[0].slice.list, c[1].slice.list, c[2].slice.list}
//}

//func (c RChanN) ReadDelta(Δstart, Δstop int) [3][]float32 {
//	var next [3][]float32
//	for i := range c {
//		c[i].mutex.ReadDelta(Δstart, Δstop)
//		a, b := c[i].mutex.RRange()
//		next[i] = c[i].slice.Slice(a, b).list
//	}
//	return next
//}
