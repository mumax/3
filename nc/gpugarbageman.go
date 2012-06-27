package nc

// Garbageman recycles garbage slices.

import (
	"runtime"
	"sync/atomic"
)

type GpuGarbageman struct {
	bysize   map[int]chan GpuBlock
	size     [3]int
	numAlloc int32
}

// Return a buffer of size WarpSize(), 
// recycle an old one if possible.
// Buffers created in this way should be Recyle()d
// when not used anymore, i.e., if not Send() elsewhere.
func (g *GpuGarbageman) Get() GpuBlock {
	return g.GetSize(g.size)
}

func (m *GpuGarbageman) GetSize(size [3]int) GpuBlock {
	if _, ok := m.bysize[prod(size)]; !ok {
		m.bysize[prod(size)] = make(chan GpuBlock, BUFSIZE*NumWarp())
		Debug("Now recycling slices of size", prod(size))
	}
	chute := m.bysize[prod(size)]

	runtime.Gosched() // good idea? give others the chance to recycle first.
	select {
	case buffer := <-chute:
		Assert(buffer.Len() == prod(size))
		buffer.size = size
		return buffer
	default:
		return m.Alloc(size) // TODO: if alloc < maxalloc
	}
	panic("bug") // unreachable
	var b GpuBlock
	return b
}

func prod(size [3]int) int { return size[0] * size[1] * size[2] }

func (g *GpuGarbageman) N() int {
	return g.size[0] * g.size[1] * g.size[2]
}

// Return a freshly allocated & managed block.
func (g *GpuGarbageman) Alloc(size [3]int) GpuBlock {
	slice := MakeGpuBlock(size)
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
		chute := m.bysize[g.Len()]
		if g.refcount.Load() == 0 {
			select {
			case chute <- g: //Debug("recycling", g)
			default:
				Debug("spilling", g)
				g.Free()
			}
		} else { // cannot be recycled, just yet
			g.refcount.Add(-1)
		}
	}
}

func (g *GpuGarbageman) Init(warpSize [3]int) {
	g.bysize = make(map[int]chan GpuBlock)
	g.size = warpSize
}
