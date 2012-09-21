package gpu

import (
	"github.com/barnex/cuda4/safe"
	"nimble-cube/core"
)

type Chan struct {
	safe.Float32s
	core.RWMutex
}

func MakeChan(size [3]int) *Chan {
	c := new(Chan)
	c.Init(size)
	return c
}

func (c *Chan) Init(size [3]int) {
	c.Float32s = safe.MakeFloat32s(core.Prod(size))
	c.RWMutex.Init(core.Prod(size))
}

type RChan struct {
	safe.Float32s
	*core.RMutex
}

func (c *Chan) ReadOnly() RChan {
	return RChan{c.Float32s, c.RWMutex.NewReader()}
}
