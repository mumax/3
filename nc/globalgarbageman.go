package nc

import (
	"sync/atomic"
)

// Global garbageman.
var garbageman Garbageman

// Get a buffer from the global garbageman.
func Buffer() Block {
	return garbageman.Get()
}

func Buffer3() [3]Block {
	return [3]Block{Buffer(), Buffer(), Buffer()}
}

// Recyle a buffer from the global garbageman.
func Recycle(g ...Block) { garbageman.Recycle(g...) }

func Recycle3(garbages ...[3]Block) {
	for _, g := range garbages {
		for _, c := range g {
			garbageman.Recycle(c)
		}
	}
}

func NumAlloc() int { return int(atomic.LoadInt32(&garbageman.numAlloc)) }

// Global garbageman.
var gpugarbageman GpuGarbageman

// Get a buffer form the global garbageman.
func GpuBuffer() GpuBlock {
	return gpugarbageman.Get()
}

func RecycleGpu(g ...GpuBlock) {
	gpugarbageman.Recycle(g...)
}

func NumGpuAlloc() int {
	return int(atomic.LoadInt32(&gpugarbageman.numAlloc))
}

func InitGarbageman() {
	// recycling buffer may be huge, it should not waste any memory.
	garbageman.Init(WarpSize())
	gpugarbageman.Init(WarpSize())
}

// Garbage chute buffer size (blocks)
const BUFSIZE = 1000
