package gpu

import (
	"github.com/barnex/cuda4/safe"
	"nimble-cube/core"
)

type chandata struct {
	list safe.Float32s
	size [3]int
}

func (c *chandata) Size() [3]int              { return c.size }
func (c *chandata) UnsafeData() safe.Float32s { return c.list }

type Chan struct {
	chandata
	mutex *core.RWMutex
}

func MakeChan(size [3]int, tag string) Chan {
	return Chan{chandata{safe.MakeFloat32s(core.Prod(size)), size}, core.NewRWMutex(core.Prod(size), tag)}
}

// WriteNext locks and returns a slice of length n for 
// writing the next n elements to the Chan.
// When done, WriteDone() should be called to "send" the
// slice down the Chan. After that, the slice is not valid any more.
func (c *Chan) WriteNext(n int) safe.Float32s {
	c.mutex.WriteNext(n)
	a, b := c.mutex.WRange()
	return c.list.Slice(a, b)
}

// WriteDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan.
func (c *Chan) WriteDone() {
	c.mutex.WriteDone()
}
