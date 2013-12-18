package cuda

import (
	"github.com/barnex/cuda5/cu"
	"github.com/mumax/3/data"
	"log"
	"unsafe"
)

var (
	buf_pool  map[int][]unsafe.Pointer    // maps buffer size to pool
	buf_check map[unsafe.Pointer]struct{} // check if pointer originates here
)

const buf_max = 100 // maximum number of buffers to allocate

// Returns a GPU slice for temporary use. To be returned to the pool with Recycle
func Buffer(nComp int, size [3]int) *data.Slice {
	if buf_pool == nil {
		buf_pool = make(map[int][]unsafe.Pointer)
		buf_check = make(map[unsafe.Pointer]struct{})
	}

	N := prod(size)
	pool := buf_pool[N]
	nFromPool := iMin(nComp, len(pool))
	ptrs := make([]unsafe.Pointer, nComp)

	for i := 0; i < nFromPool; i++ {
		ptrs[i] = pool[len(pool)-i-1]
	}
	buf_pool[N] = pool[:len(pool)-nFromPool]

	for i := nFromPool; i < nComp; i++ {
		if len(buf_check) >= buf_max {
			log.Panic("too many buffers in use, possible memory leak")
		}
		ptrs[i] = MemAlloc(int64(cu.SIZEOF_FLOAT32 * N))
		buf_check[ptrs[i]] = struct{}{} // mark this pointer as mine
	}
	return data.SliceFromPtrs(size, data.GPUMemory, ptrs)
}

// Returns a buffer obtained from GetBuffer to the pool.
func Recycle(s *data.Slice) {
	N := s.Len()
	pool := buf_pool[N]
	for i := 0; i < s.NComp(); i++ {
		ptr := s.DevPtr(i)
		if _, ok := buf_check[ptr]; !ok {
			log.Panic("recyle: was not obtained with getbuffer")
		}
		pool = append(pool, ptr)
	}
	s.Disable() // make it unusable, protect against accidental use after recycle
	buf_pool[N] = pool
}

func FreeBuffers() {
	for _, size := range buf_pool {
		for i := range size {
			cu.DevicePtr(size[i]).Free()
			size[i] = nil
		}
	}
	buf_pool = make(map[int][]unsafe.Pointer)
	buf_check = make(map[unsafe.Pointer]struct{})
}
