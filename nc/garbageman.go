package nc

// Garbageman recycles garbage slices.

import (
	"github.com/barnex/cuda4/cu"
	"sync/atomic"
)

// Global garbageman.
var garbageman Garbageman

// Get a buffer form the global garbageman.
func Buffer()Block{return garbageman.Get()}

type Garbageman struct{
	recycled chan Block
	numAlloc int32
}

// Return a buffer, recycle an old one if possible.
// Buffers created in this way should be Recyle()d
// when not used anymore, i.e., if not Send() elsewhere.
func (g*Garbageman)Get() Block{
	select{
		case b:=<-g.recycled: return b
		default: return g.Alloc() // TODO: if alloc < maxalloc
	}
	panic("bug") // unreachable
	return g.Alloc()
}

// Return a freshly allocated & managed block.
func(g*Garbageman)Alloc()Block{
	slice := MakeBlock(WarpSize())
	if *flag_pagelock {
		SetCudaCtx()
		cu.MemHostRegister(slice.UnsafePointer(), slice.Bytes(), cu.MEMHOSTREGISTER_PORTABLE)
	}
	atomic.AddInt32(&g.numAlloc, 1)
	slice.Refcount = new(Refcount)
	return slice
}


// Recycle the block.
func (m*Garbageman)Recycle(garbages ...Block) {
	for _, g := range garbages {
		if g.Refcount == nil {
			continue // slice does not originate from here
		}
		if g.Count() == 0{
			select{
				case m.recycled<-g:
				default: Debug("spilling", g)
			}
		} else { // cannot be recycled, just yet
			g.Increment(-1)
		}
	}
}

//func Recycle3(garbages ...[3]Block) {
//	for _, g := range garbages {
//		Recycle(g[X], g[Y], g[Z])
//	}
//}
