package gpu

import (
	"github.com/barnex/cuda5/safe"
	"nimble-cube/core"
)

type RChan struct {
	chandata
	mutex *core.RMutex
}

func (c *Chan1) MakeRChan() RChan {
	return RChan{c.chandata, c.mutex.MakeRMutex()}
}

// ReadNext locks and returns a slice of length n for 
// reading the next n elements from the Chan.
// When done, ReadDone() should be called .
// After that, the slice is not valid any more.
func (c *RChan) ReadNext(n int) safe.Float32s {
	c.mutex.ReadNext(n)
	a, b := c.mutex.RRange()
	return c.list.Slice(a, b)
}

// ReadDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan.
func (c *RChan) ReadDone() {
	c.mutex.ReadDone()
}
