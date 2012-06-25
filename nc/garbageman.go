package nc

// Garbageman recycles garbage slices.

import (
	"github.com/barnex/cuda4/cu"
	"sync/atomic"
)

var (
	recycled              stack
	gpuRecycled           gpuStack
	NumAlloc, NumGpuAlloc int
)

// increment the reference count by count.
func incr(s Block, count int32) {
	if s.refcount != nil {
		atomic.AddInt32(s.refcount, count)
	}
}

func incrGpu(s GpuBlock, count int32) {
	if s.refcount != nil {
		atomic.AddInt32(s.refcount, count)
	}
}

// increment the reference count by count.
func incr3(s [3]Block, count int32) {
	for i := range s {
		incr(s[i], count)
	}
}

// See Buffer()
func Buffer3() [3]Block {
	return [3]Block{Buffer(), Buffer(), Buffer()}
}

// See GpuBuffer()
func GpuBuffer3() [3]GpuBlock {
	return [3]GpuBlock{GpuBuffer(), GpuBuffer(), GpuBuffer()}
}

// Return a buffer, recycle an old one if possible.
// Buffers created in this way should be Recyle()d
// when not used anymore, i.e., if not Send() elsewhere.
func Buffer() Block {
	if f := recycled.pop(); !f.IsNil() {
		return f
	}
	slice := MakeBlock(WarpSize())

	if *flag_pagelock {
		SetCudaCtx()
		cu.MemHostRegister(slice.UnsafePointer(),
			slice.Bytes(),
			cu.MEMHOSTREGISTER_PORTABLE)
	}

	NumAlloc++
	slice.refcount = new(int32)
	return slice
}

func GpuBuffer() GpuBlock {
	if f := gpuRecycled.pop(); f.Pointer() != 0 {
		Assert(f.N() == WarpLen())
		f.size = WarpSize()
		return f
	}
	slice := MakeGpuBlock(WarpSize())
	NumGpuAlloc++
	slice.refcount = new(int32)
	return slice
}

func Recycle(garbages ...Block) {
	for _, g := range garbages {
		if g.refcount == nil {
			continue // slice does not originate from here
		}
		if atomic.LoadInt32(g.refcount) == 0 { // can be recycled
			recycled.push(g)
		} else { // cannot be recycled, just yet
			atomic.AddInt32(g.refcount, -1)
		}
	}
}

func RecycleGpu(garbages ...GpuBlock) {
	for _, g := range garbages {
		if g.refcount == nil {
			continue // slice does not originate from here
		}
		if atomic.LoadInt32(g.refcount) == 0 { // can be recycled
			gpuRecycled.push(g)
		} else { // cannot be recycled, just yet
			atomic.AddInt32(g.refcount, -1)
		}
	}
}

func Recycle3(garbages ...[3]Block) {
	for _, g := range garbages {
		Recycle(g[X], g[Y], g[Z])
	}
}
