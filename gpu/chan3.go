package gpu

import (
	"github.com/barnex/cuda4/safe"
	"nimble-cube/core"
)

type chan3data struct {
	list [3]safe.Float32s
	size [3]int
}

func (c *chan3data) Size() [3]int { return c.size }

func (c *chan3data) UnsafeData() [3]safe.Float32s { return c.list }

type Chan3 struct {
	chan3data
	mutex *core.RWMutex
}

func MakeChan3(size [3]int, tag string) Chan3 {
	return Chan3{chan3data{MakeVectors(core.Prod(size)), size}, core.NewRWMutex(core.Prod(size), tag)}
}

// WriteNext locks and returns a slice of length n for 
// writing the next n elements to the Chan3.
// When done, WriteDone() should be called to "send" the
// slice down the Chan3. After that, the slice is not valid any more.
func (c *Chan3) WriteNext(n int) [3]safe.Float32s {
	c.mutex.WriteNext(n)
	a, b := c.mutex.WRange()
	return [3]safe.Float32s{c.list[0].Slice(a, b), c.list[1].Slice(a, b), c.list[2].Slice(a, b)}
}

// WriteDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan3.
func (c *Chan3) WriteDone() {
	c.mutex.WriteDone()
}
