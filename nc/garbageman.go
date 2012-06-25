package nc

// Garbageman recycles garbage slices.

import (
	"github.com/barnex/cuda4/cu"
	"sync/atomic"
)

type Garbageman struct {
	recycled chan Block
	size     [3]int
	numAlloc int32
}

// Return a buffer, recycle an old one if possible.
// Buffers created in this way should be Recyle()d
// when not used anymore, i.e., if not Send() elsewhere.
func (g *Garbageman) Get() Block {
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
func (g *Garbageman) Alloc() Block {
	slice := MakeBlock(g.size)
	if *flag_pagelock {
		SetCudaCtx()
		cu.MemHostRegister(slice.UnsafePointer(), slice.Bytes(), cu.MEMHOSTREGISTER_PORTABLE)
	}
	atomic.AddInt32(&g.numAlloc, 1)
	slice.refcount = new(Refcount)
	return slice
}

// Recycle the block.
func (m *Garbageman) Recycle(garbages ...Block) {
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
			}
		} else { // cannot be recycled, just yet
			g.refcount.Add(-1)
		}
	}
}

func (g *Garbageman) Init(warpSize [3]int) {
	g.recycled = make(chan Block, BUFSIZE*NumWarp())
	g.size = warpSize
}
