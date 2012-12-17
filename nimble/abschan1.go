package nimble

import (
	"code.google.com/p/mx3/core"
)

type chan1 struct {
	Info
	buffer Slice
	mutex
}

func newchan1(tag, unit string, m *Mesh, memType MemType, bufBlocks int) *chan1 {
	N := -666
	if bufBlocks < 1 { // means auto
		N = m.NCell() // buffer all
	} else {
		N = m.BlockLen() * bufBlocks
	}
	return aschan1(MakeSlice(N, memType), tag, unit, m, newRWMutex(N))
}

func aschan1(buffer Slice, tag, unit string, mesh *Mesh, lock mutex) *chan1 {
	core.AddQuant(tag)
	info := newInfo(tag, unit, mesh)
	return &chan1{info, buffer, lock}
}

func (c chan1) MemType() MemType { return c.buffer.MemType }

func (c chan1) UnsafeData() Slice {
	if c.isLocked() {
		panic("unsafedata: mutex is locked")
	}
	return c.buffer
}

func (c chan1) UnsafeArray() [][][]float32 {
	if c.isLocked() {
		panic("unsafearray: mutex is locked")
	}
	return core.Reshape(c.UnsafeData().Host(), c.Mesh.Size())
}

// WriteDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan.
//func (c chan1) done() { c.unlock() }

// WriteNext returns a buffer Slice of length n to which data
// can be written. Should be followed by ReadDone().
func (c chan1) next(n int) Slice {
	c.lockNext(n)
	a, b := c.mutex.lockedRange()
	return c.buffer.Slice(a, b)
}

// NComp returns the number of components (1: scalar, 3: vector, ...)
func (c chan1) NComp() int { return 1 }

// BufLen returns the largest buffer size n that can be obained
// with ReadNext/WriteNext.
func (c chan1) BufLen() int { return c.buffer.Len() }

func (c chan1) NBufferedBlocks() int { return idiv(c.NCell(), c.buffer.Len()) }

//func (c *chan1) WriteDelta(Δstart, Δstop int) []float32 {
//	c.mutex.WriteDelta(Δstart, Δstop)
//	a, b := c.mutex.WRange()
//	return c.slice.list[a:b]
//}

// safe integer division.
func idiv(a, b int) int {
	core.Assert(a%b == 0)
	return a / b
}
