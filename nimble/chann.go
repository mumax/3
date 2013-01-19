package nimble

import (
//"code.google.com/p/mx3/core"
)

// ChanN is a Chan that passes N-component data.
type chanN struct {
	buffer Slice
	lock   [MAX_COMP]mutex
}

type ChanN struct {
	chanN
}

type RChanN struct {
	chanN
	next Slice // to avoid allocation
}

func MakeChanN(nComp int, tag, unit string, m *Mesh, memType MemType, bufBlocks int) ChanN {
	N := bufferSize(m, bufBlocks)
	buffer := makeSliceN(nComp, N, m, memType)
	var lock [MAX_COMP]mutex
	for c := 0; c < nComp; c++ {
		lock[c] = newRWMutex(N)
	}
	return ChanN{chanN{buffer, lock}}
}

func (c ChanN) NComp() int  { return int(c.buffer.nComp) }
func (c ChanN) BufLen() int { return c.buffer.Len() } //?
//func (c ChanN) NBufferedBlocks() int { return c.comp[0].NBufferedBlocks() }
func (c ChanN) MemType() MemType { return c.buffer.MemType }

//func (c ChanN) ChanN() ChanN     { return c } // implements Chan iface
//func (c ChanN) Comp(i int) Chan1 { return c.comp[i] }

func (c ChanN) UnsafeData() Slice {
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
func (c ChanN) WriteNext(n int) Slice {
	c.lock[0].lockNext(n)
	a, b := c.comp[i].lockedRange()

	return next
}

// WriteDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan3.
func (c ChanN) WriteDone() {
	for i := range c.comp {
		c.comp[i].WriteDone()
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
