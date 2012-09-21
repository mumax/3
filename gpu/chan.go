package gpu

import (
	"github.com/barnex/cuda4/safe"
	"nimble-cube/core"
)

type Chan struct {
	safe.Float32s
	*core.RWMutex
}

func MakeChan(size [3]int) Chan {
	return Chan{safe.MakeFloat32s(core.Prod(size)), core.NewRWMutex(core.Prod(size))}
}

type RChan struct {
	safe.Float32s
	*core.RMutex
}

func (c *Chan) ReadOnly() RChan {
	return RChan{c.Float32s, c.RWMutex.NewReader()}
}
