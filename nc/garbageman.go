package nc

// Garbageman recycles garbage slices.

import (
	"github.com/barnex/cuda4/cu"
	"sync"
)

var (
	recycled              stack
	gpuRecycled           gpuStack
	refcount              = make(map[*float32]int)
	gpuRefcount           = make(map[GpuBlock]int)
	NumAlloc, NumGpuAlloc int
	cpulock, gpulock      sync.Mutex
)

// increment the reference count by count.
func incr(s Block, count int) {
	cpulock.Lock()
	if prev, ok := refcount[s.Pointer()]; ok {
		refcount[s.Pointer()] = prev + count
	}
	cpulock.Unlock()
	//Assert(len(refcount) == NumAlloc)
}

func incrGpu(s GpuBlock, count int) {
	//Debug("incrgpu", s, count)
	gpulock.Lock()
	if prev, ok := gpuRefcount[s]; ok {
		gpuRefcount[s] = prev + count
	}
	gpulock.Unlock()
	Assert(len(gpuRefcount) == NumGpuAlloc)
}

// increment the reference count by count.
func incr3(s [3]Block, count int) {
	cpulock.Lock()
	for c := 0; c < 3; c++ {
		if prev, ok := refcount[s[c].Pointer()]; ok {
			refcount[s[c].Pointer()] = prev + count
		}
	}
	cpulock.Unlock()
}

// Return a buffer, recycle an old one if possible.
// Buffers created in this way should be Recyle()d
// when not used anymore, i.e., if not Send() elsewhere.
func Buffer() Block {
	cpulock.Lock()
	b := buffer()
	cpulock.Unlock()
	return b
}

func GpuBuffer() GpuBlock {
	gpulock.Lock()
	b := gpuBuffer()
	gpulock.Unlock()
	return b
}

// See Buffer()
func Buffer3() [3]Block {
	cpulock.Lock()
	b := [3]Block{buffer(), buffer(), buffer()}
	cpulock.Unlock()
	return b
}

func GpuBuffer3() [3]GpuBlock {
	gpulock.Lock()
	b := [3]GpuBlock{gpuBuffer(), gpuBuffer(), gpuBuffer()}
	gpulock.Unlock()
	return b
}

func assureCtx() {
	if cu.CtxGetCurrent() != cudaCtx {
		cudaCtx.SetCurrent()
	}
}

// not synchronized.
func buffer() Block {
	if f := recycled.pop(); !f.IsNil() {
		//log.Println("re-use", &f[0])
		return f
	}
	slice := MakeBlock(WarpSize())

	if *flag_pagelock {
		assureCtx()
		// runtime.Pray("please don't swap my OS thread right now, dear Dymitry!")
		cu.MemHostRegister(slice.UnsafePointer(),
			slice.Bytes(),
			cu.MEMHOSTREGISTER_PORTABLE)
	}

	NumAlloc++
	//log.Println("alloc", &slice[0])
	refcount[slice.Pointer()] = 0
	return slice
}

func gpuBuffer() GpuBlock {
	if f := gpuRecycled.pop(); f.Pointer() != 0 {
		//Debug("re-use", f)
		return f
	}
	slice := MakeGpuBlock(WarpSize())
	NumGpuAlloc++
	//Debug("alloc", slice)
	gpuRefcount[slice] = 0
	return slice
}

func Recycle(garbages ...Block) {
	cpulock.Lock()

	for _, g := range garbages {
		count, ok := refcount[g.Pointer()]
		if !ok {
			//log.Println("skipping", &g[0])
			continue // slice does not originate from here
		}
		if count == 0 { // can be recycled
			recycled.push(g)
			//log.Println("spilling", &g[0])
			//delete(refcount, &g[0]) // allow it to be GC'd TODO: spilltest
		} else { // cannot be recycled, just yet
			//log.Println("decrementing", &g[0], ":", count-1)
			refcount[g.Pointer()] = count - 1
		}

	}
	cpulock.Unlock()
}

func RecycleGpu(garbages ...GpuBlock) {
	gpulock.Lock()

	for _, g := range garbages {
		count, ok := gpuRefcount[g]
		if !ok {
			//Debug("skipping", g)
			continue // slice does not originate from here
		}
		if count == 0 { // can be recycled
			gpuRecycled.push(g)
			//Debug("recycling", g)
		} else { // cannot be recycled, just yet
			//Debug("decrementing", g, ":", count-1)
			gpuRefcount[g] = count - 1
		}

	}
	gpulock.Unlock()
}

func Recycle3(garbages ...[3]Block) {
	for _, g := range garbages {
		Recycle(g[X], g[Y], g[Z])
	}
}
