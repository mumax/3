package cuda

import (
	"github.com/barnex/cuda5/cu"
	"github.com/mumax/3/data"
	"log"
	"sync"
	"unsafe"
)

var (
	buf_lock  sync.Mutex
	buf_pool  map[int][]unsafe.Pointer    // maps buffer size to pool
	buf_check map[unsafe.Pointer]struct{} // check if pointer originates here
	buf_count int                         // total allocated buffers
)

const buf_max = 100 // maximum number of buffers to allocate

// Returns a GPU slice for temporary use. To be returned to the pool with Recycle
func Buffer(nComp int, m *data.Mesh) *data.Slice {
	buf_lock.Lock()
	defer buf_lock.Unlock()

	if buf_pool == nil {
		buf_pool = make(map[int][]unsafe.Pointer)
		buf_check = make(map[unsafe.Pointer]struct{})
	}

	N := m.NCell()
	pool := buf_pool[N]
	nFromPool := iMin(nComp, len(pool))
	ptrs := make([]unsafe.Pointer, nComp)

	for i := 0; i < nFromPool; i++ {
		ptrs[i] = pool[len(pool)-i-1]
	}
	buf_pool[N] = pool[:len(pool)-nFromPool]

	for i := nFromPool; i < nComp; i++ {
		log.Println("cuda: alloc buffer")
		if buf_count >= buf_max {
			log.Panic("too many buffers in use, possible memory leak")
		}
		buf_count++
		ptrs[i] = MemAlloc(int64(cu.SIZEOF_FLOAT32 * N))
		buf_check[ptrs[i]] = struct{}{} // mark this pointer as mine
	}
	return data.SliceFromPtrs(m, data.GPUMemory, ptrs)
}

// Returns a buffer obtained from GetBuffer to the pool.
func Recycle(s *data.Slice) {
	buf_lock.Lock()
	defer buf_lock.Unlock()

	N := s.Mesh().NCell()
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
