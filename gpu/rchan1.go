package gpu

import (
	"github.com/barnex/cuda5/safe"
	"nimble-cube/core"
)

type RChan1 struct {
	chandata
	mutex *core.RMutex
}

func (c *Chan1) MakeRChan() RChan1 {
	return RChan1{c.chandata, c.mutex.MakeRMutex()}
}

// ReadNext locks and returns a slice of length n for 
// reading the next n elements from the Chan.
// When done, ReadDone() should be called .
// After that, the slice is not valid any more.
func (c *RChan1) ReadNext(n int) safe.Float32s {
	c.mutex.ReadNext(n)
	a, b := c.mutex.RRange()
	return c.list.Slice(a, b)
}

// ReadDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan.
func (c *RChan1) ReadDone() {
	c.mutex.ReadDone()
}
