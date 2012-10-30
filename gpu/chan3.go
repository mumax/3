package gpu

import (
	"github.com/barnex/cuda5/safe"
	"nimble-cube/core"
)

// Chan of 3-vector data.
type Chan3 [3]Chan1

func MakeChan3(tag, unit string, m *core.Mesh, blocks ...int) Chan3 {
	//tag = core.UniqueTag(tag)
	core.AddQuant(tag)
	var c Chan3
	for i := range c {
		c[i] = MakeChan1(tag, unit, m, blocks...)
	}
	return c
}

func HostChan3(tag, unit string, m *core.Mesh, blocks ...int) Chan3 {
	//tag = core.UniqueTag(tag)
	var c Chan3
	for i := range c {
		c[i] = HostChan1(tag, unit, m, blocks...)
	}
	return c
}

func (c Chan3) ChanN() ChanN { return ChanN(c[:]) }

// UnsafeData returns the underlying storage without locking.
// Intended only for page-locking, not for reading or writing.
func (c *Chan3) UnsafeData() [3]safe.Float32s {
	return [3]safe.Float32s{c[0].list, c[1].list, c[2].list}
}

func (c *Chan3) Mesh() *core.Mesh { return c[0].Mesh }
func (c *Chan3) Size() [3]int     { return c[0].Size() }
func (c *Chan3) Unit() string     { return c[0].Unit() }
func (c *Chan3) Tag() string      { return c[0].Tag() }

// WriteNext locks and returns a slice of length n for 
// writing the next n elements to the Chan3.
// When done, WriteDone() should be called to "send" the
// slice down the Chan3. After that, the slice is not valid any more.
func (c *Chan3) WriteNext(n int) [3]safe.Float32s {
	var next [3]safe.Float32s
	for i := range c {
		c[i].WriteNext(n)
		a, b := c[i].mutex.WRange()
		next[i] = c[i].list.Slice(a, b)
	}
	return next
}

// WriteDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan3.
func (c *Chan3) WriteDone() {
	for i := range c {
		c[i].WriteDone()
	}
}
