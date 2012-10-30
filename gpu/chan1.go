package gpu

import (
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"nimble-cube/core"
)

type chandata struct {
	list safe.Float32s
	*core.Info
}

func (c *chandata) UnsafeData() safe.Float32s { return c.list }

type Chan1 struct {
	chandata
	mutex *core.RWMutex
}

func MakeChan1(tag, unit string, m *core.Mesh, blocks ...int) Chan1 {
	//tag = core.UniqueTag(tag)
	core.AddQuant(tag)
	info := core.NewInfo(tag, unit, m, blocks...)
	len_ := info.BlockLen()
	return Chan1{chandata{safe.MakeFloat32s(len_), info}, core.NewRWMutex(len_, tag)}
}

func HostChan1(tag, unit string, m *core.Mesh, blocks ...int) Chan1 {
	//tag = core.UniqueTag(tag)
	info := core.NewInfo(tag, unit, m, blocks...)
	len_ := info.BlockLen()
	return Chan1{chandata{MakeHostFloat32s(len_), info}, core.NewRWMutex(len_, tag)}
}

func MakeHostFloat32s(len_ int) safe.Float32s {
	var storage safe.Float32s
	bytes := int64(len_) * cu.SIZEOF_FLOAT32
	ptr := cu.MemAllocHost(bytes)
	cap_ := len_
	storage.UnsafeSet(ptr, len_, cap_)
	return storage
}

// WriteNext locks and returns a slice of length n for 
// writing the next n elements to the Chan.
// When done, WriteDone() should be called to "send" the
// slice down the Chan. After that, the slice is not valid any more.
func (c Chan1) WriteNext(n int) safe.Float32s {
	c.mutex.WriteNext(n)
	a, b := c.mutex.WRange()
	return c.list.Slice(a, b)
}

// WriteDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan.
func (c Chan1) WriteDone() {
	c.mutex.WriteDone()
}
