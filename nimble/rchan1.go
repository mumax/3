package nimble

// TODO: can be completely absorbed in Chan1
// if we use mutex interface Next(), Done()

import (
	"code.google.com/p/mx3/core"
)

// Read-only Chan.
type RChan1 struct {
	*Info
	buffer Slice
	mutex  *rMutex
}

func (c Chan1) NewReader() RChan1 {
	return RChan1{c.Info, c.buffer, c.mutex.MakeRMutex()}
}

func (c RChan1) UnsafeData() Slice {
	if c.mutex.rw.isLocked() {
		panic("unsafearray: mutex is locked")
	}
	return c.buffer
}
func (c RChan1) UnsafeArray() [][][]float32 {
	return core.Reshape(c.UnsafeData().Host(), c.Mesh.Size())
}

// ReadNext locks and returns a slice of length n for
// reading the next n elements from the Chan.
// When done, ReadDone() should be called .
// After that, the slice is not valid any more.
func (c RChan1) ReadNext(n int) Slice {
	c.mutex.ReadNext(n)
	a, b := c.mutex.RRange()
	return c.buffer.Slice(a, b)
}

// ReadDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan.
func (c RChan1) ReadDone() {
	c.mutex.ReadDone()
}

//func (c *RChan1) ReadDelta(Δstart, Δstop int) []float32 {
//	c.mutex.ReadDelta(Δstart, Δstop)
//	a, b := c.mutex.RRange()
//	return c.slice.Slice(a, b).list
//}
