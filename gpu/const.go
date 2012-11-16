package gpu

import (
 "fmt"
"code.google.com/p/nimble-cube/nimble"
)

// Const represents a value that is constant in time and space (homogeneous).
type Const struct {
	output nimble.ChanN
}

func NewConst(tag, unit string, m *nimble.Mesh, value ...float64) *Const {
	nComp := len(value)
	if nComp < 1{
		panic(fmt.Errorf("newconst: need at least one value"))
	}
	c := new(Const)
	data := nimble.MakeSlices(nComp, m.BlockLen(), nimble.GPUMemory)
	for i:=range data{
		data[i].Device().Memset(float32(value[i]))
	}
	c.output = nimble.AsChan(data, tag, unit, m)
	nimble.Stack(c)
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
