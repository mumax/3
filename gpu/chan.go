package gpu

import (
	"github.com/barnex/cuda4/safe"
	"nimble-cube/core"
)

type Chan struct {
	list  safe.Float32s
	mutex *core.RWMutex
}

func MakeChan(size [3]int) Chan {
	return Chan{safe.MakeFloat32s(core.Prod(size)), core.NewRWMutex(core.Prod(size))}
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
