package gpu

import (
	"github.com/barnex/cuda5/safe"
	"nimble-cube/core"
)

type RChan struct {
	chandata
	mutex *core.RMutex
}

func (c *Chan) MakeRChan() RChan {
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

//type Chan3 [3]Chan
//
//func MakeChan3(size [3]int) Chan3 {
//	return Chan3{MakeChan(size), MakeChan(size), MakeChan(size)}
//}
//
//func (c *Chan3) Vectors() [3]safe.Float32s {
//	return [3]safe.Float32s{c[0].Float32s, c[1].Float32s, c[2].Float32s}
//}
//
//func (c *Chan3) RWMutex() core.RWMutex3 {
//	return core.RWMutex3{c[0].RWMutex, c[1].RWMutex, c[2].RWMutex}
//}
//
//// Read-only Chan3
//type RChan3 [3]RChan
//
//func (c *Chan3) ReadOnly() RChan3 {
//	return RChan3{c[0].ReadOnly(), c[1].ReadOnly(), c[2].ReadOnly()}
//}
