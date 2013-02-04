package nimble

import (
	"code.google.com/p/mx3/core"
)

type chanN struct {
	buffer    Slice
	lock      [MAX_COMP]mutex
	tag, unit string
	mesh      *Mesh
}

// ChanN is a Chan that passes N-component data. R/W.
// TODO: Chan
type ChanN struct {
	chanN
}

// Read-only.
// TODO: Reader
type RChanN struct {
	chanN
	next Slice // to avoid allocation
}

func MakeChanN(nComp int, tag, unit string, m *Mesh, memType MemType, bufBlocks int) ChanN {
	N := bufferSize(m, bufBlocks)
	buffer := makeSlice(nComp, N, memType)
	var lock [MAX_COMP]mutex
	for c := 0; c < nComp; c++ {
		lock[c] = newRWMutex(N)
	}
	return ChanN{chanN{buffer, lock, "", "", m}}
}

// NComp returns the number of components.
// 	scalar: 1
// 	vector: 3
// 	...
func (c *chanN) NComp() int {
	return int(c.buffer.nComp)
}

// BufLen returns the number of buffered elements.
// This is the largest number of elements that can be read/written at once.
func (c *chanN) BufLen() int {
	return c.buffer.Len()
} //?

//func (c ChanN) NBufferedBlocks() int { return c.comp[0].NBufferedBlocks() }
//func (c ChanN) MemType() MemType { return c.buffer.MemType }
//func (c ChanN) ChanN() ChanN     { return c } // implements Chan iface
//func (c ChanN) Comp(i int) Chan1 { return c.comp[i] }

// UnsafeData returns the data buffer without locking.
// To be used with extreme care.
func (c *chanN) UnsafeData() Slice {
	return c.buffer
}

//func (c ChanN) UnsafeArray() [][][][]float32 {
//	s := make([][][][]float32, c.NComp())
//	for i := range s {
//		s[i] = c.comp[i].UnsafeArray()
//	}
//	return s
//}

//func (c ChanN) Chan3() Chan3 {
//	core.Assert(c.NComp() == 3)
//	return Chan3{c.comp}
//}

//func (c ChanN) Chan1() Chan1 {
//	core.Assert(c.NComp() == 1)
//	return c.comp[0]
//}

// WriteNext locks and returns a slice of length n for
// writing the next n elements to the Chan3.
// When done, WriteDone() should be called to "send" the
// slice down the Chan3. After that, the slice is not valid any more.
func (c *chanN) next(n int) Slice {
	c.lock[0].lockNext(n)
	a, b := c.lock[0].lockedRange()
	for i := 1; i < len(c.lock); i++ {
		c.lock[i].lockNext(n)
		α, β := c.lock[i].lockedRange()
		if α != a || β != b {
			panic("chan: next: inconsistent state")
		}
	}
	next := c.buffer.Slice(a, b)
	return next
}

func (c *chanN) done() {
	for i := range c.lock {
		c.lock[i].unlock()
	}
}

func (c *ChanN) WriteNext(n int) Slice {
	return c.chanN.next(n)
}

func (c *ChanN) WriteDone() {
	c.chanN.done()
}

func (c *RChanN) ReadNext(n int) Slice {
	return c.chanN.next(n)
}

func (c *RChanN) ReadDone() {
	c.chanN.done()
}

// WriteDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan3.

//func (c ChanN) WriteDelta(Δstart, Δstop int) [][]float32 {
//	next := make([][]float32, c.NComp())
//	for i := range c {
//		c[i].WriteDelta(Δstart, Δstop)
//		a, b := c[i].mutex.WRange()
//		next[i] = c[i].slice.Slice(a, b).list
//	}
//	return next
//}

func bufferSize(m *Mesh, bufBlocks int) int {
	N := -666
	if bufBlocks < 1 { // means auto
		N = m.NCell() // buffer all
	} else {
		N = m.BlockLen() * bufBlocks
	}
	return N
}

// safe integer division.
func idiv(a, b int) int {
	core.Assert(a%b == 0)
	return a / b
}
