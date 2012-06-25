package nc

// Garbageman recycles garbage slices.

import (
	"sync/atomic"
)

type GpuGarbageman struct {
	recycled chan GpuBlock
	size     [3]int
	numAlloc int32
}

// Return a buffer, recycle an old one if possible.
// Buffers created in this way should be Recyle()d
// when not used anymore, i.e., if not Send() elsewhere.
func (g *GpuGarbageman) Get() GpuBlock {
	select {
	case b := <-g.recycled:
		return b
	default:
		return g.Alloc() // TODO: if alloc < maxalloc
	}
	panic("bug") // unreachable
	return g.Alloc()
}

// Return a freshly allocated & managed block.
func (g *GpuGarbageman) Alloc() GpuBlock {
	slice := MakeGpuBlock(g.size)
	atomic.AddInt32(&g.numAlloc, 1)
	slice.refcount = new(Refcount)
	return slice
}

// Recycle the block.
func (m *GpuGarbageman) Recycle(garbages ...GpuBlock) {
	for _, g := range garbages {
		Assert(g.Size() == m.size)
		if g.refcount == nil {
			continue // slice does not originate from here
		}
		if g.refcount.Load() == 0 {
			select {
			case m.recycled <- g: //Debug("recycling", g)
			default:
				Debug("spilling", g)
				g.Free()
			}
		} else { // cannot be recycled, just yet
			g.refcount.Add(-1)
		}
	}
}

func (g *GpuGarbageman) Init(warpSize [3]int, buffer int) {
	g.recycled = make(chan GpuBlock, buffer)
	g.size = warpSize
}

//func Recycle3(garbages ...[3]GpuBlock) {
//	for _, g := range garbages {
//		Recycle(g[X], g[Y], g[Z])
//	}
//}

//func GpuBuffer() GpuGpuBlock {
//	if f := gpuRecycled.pop(); f.Pointer() != 0 {
//		Assert(f.N() == g.size
//		f.size = g.size
//		return f
//	}
//	slice := MakeGpuGpuBlock(WarpSize())
//	NumGpuAlloc++
//	slice.refcount = new(int32)
//	return slice
//}
