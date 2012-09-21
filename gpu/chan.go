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

type Chan3 [3]Chan

func MakeChan3(size [3]int) Chan3 {
	return Chan3{MakeChan(size), MakeChan(size), MakeChan(size)}
}

func (c *Chan3) Vectors() [3]safe.Float32s {
	return [3]safe.Float32s{c[0].Float32s, c[1].Float32s, c[2].Float32s}
}

func(c*Chan3)RWMutex()core.RWMutex3{
	return core.RWMutex3{c[0].RWMutex, c[1].RWMutex, c[2].RWMutex}
}
