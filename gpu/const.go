package gpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/nimble"
	"fmt"
)

// Const represents a value that is constant in time.
type Const struct {
	output nimble.ChanN
}

// NewConst returns a time- and space- independent constant value.
func NewConst(tag, unit string, m *nimble.Mesh, memType nimble.MemType, value []float64) *Const {
	c := newConst(len(value), tag, unit, m, memType)
	for i, data := range c.output.UnsafeData() {
		data.Device().Memset(float32(value[i]))
	}
	nimble.Stack(c) // <- !
	return c
}

// NewConstArray returns a time- independent array.
func NewConstArray(tag, unit string, m *nimble.Mesh, array [][][][]float32) *Const {
	c := newConst(len(array), tag, unit, m, nimble.GPUMemory)
	for i, data := range c.output.UnsafeData() {
		data.Device().CopyHtoD(core.Contiguous(array[i]))
	}
	nimble.Stack(c) // <- !
	return c
}

func newConst(nComp int, tag, unit string, m *nimble.Mesh, mem nimble.MemType) *Const {
	c := new(Const)
	if nComp < 1 {
		panic(fmt.Errorf("newconst: need at least one component"))
	}
	data := nimble.MakeSlices(nComp, m.BlockLen(), mem)
	c.output = nimble.AsChan(data, tag, unit, m)
	return c
}

func (c *Const) Run() {
	for {
		c.output.WriteNext(c.output.BufLen())
		c.output.WriteDone()
	}
}

func (c *Const) Output() nimble.ChanN {
	return c.output
}
