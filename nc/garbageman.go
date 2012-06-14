package nc

// Garbageman recycles garbage slices.

import (
	"sync"
)

var (
	recycled              stack
	gpuRecycled           gpuStack
	refcount              = make(map[*float32]int)
	gpuRefcount           = make(map[GpuFloats]int)
	NumAlloc, NumGpuAlloc int
	cpulock, gpulock      sync.Mutex
)

// increment the reference count by count.
func incr(s []float32, count int) {
	cpulock.Lock()
	if prev, ok := refcount[&s[0]]; ok {
		refcount[&s[0]] = prev + count
	}
	cpulock.Unlock()
	//Assert(len(refcount) == NumAlloc)
}

func incrGpu(s GpuFloats, count int) {
	//Debug("incrgpu", s, count)
	gpulock.Lock()
	if prev, ok := gpuRefcount[s]; ok {
		gpuRefcount[s] = prev + count
	}
	gpulock.Unlock()
	Assert(len(gpuRefcount) == NumGpuAlloc)
}

// increment the reference count by count.
func incr3(s [3][]float32, count int) {
	cpulock.Lock()
	for c := 0; c < 3; c++ {
		if prev, ok := refcount[&s[c][0]]; ok {
			refcount[&s[c][0]] = prev + count
		}
	}
	cpulock.Unlock()
}

// Return a buffer, recycle an old one if possible.
// Buffers created in this way should be Recyle()d
// when not used anymore, i.e., if not Send() elsewhere.
func Buffer() []float32 {
	cpulock.Lock()
	b := buffer()
	cpulock.Unlock()
	return b
}

func GpuBuffer() GpuFloats {
	gpulock.Lock()
	b := gpuBuffer()
	gpulock.Unlock()
	return b
}

// See Buffer()
func Buffer3() [3][]float32 {
	cpulock.Lock()
	b := [3][]float32{buffer(), buffer(), buffer()}
	cpulock.Unlock()
	return b
}

// not synchronized.
func buffer() []float32 {
	if f := recycled.pop(); f != nil {
		//log.Println("re-use", &f[0])
		return f
	}
	slice := make([]float32, WarpLen())
	NumAlloc++
	//log.Println("alloc", &slice[0])
	refcount[&slice[0]] = 0
	return slice
}

func gpuBuffer() GpuFloats {
	if f := gpuRecycled.pop(); f != 0 {
		//Debug("re-use", f)
		return f
	}
	slice := MakeGpuFloats(WarpLen())
	NumGpuAlloc++
	//Debug("alloc", slice)
	gpuRefcount[slice] = 0
	return slice
}

func Recycle(garbages ...[]float32) {
	cpulock.Lock()

	for _, g := range garbages {
		count, ok := refcount[&g[0]]
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
			refcount[&g[0]] = count - 1
		}

	}
	cpulock.Unlock()
}

func RecycleGpu(garbages ...GpuFloats) {
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

func Recycle3(garbages ...[3][]float32) {
	for _, g := range garbages {
		Recycle(g[X], g[Y], g[Z])
	}
}
