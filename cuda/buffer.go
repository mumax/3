package cuda

import (
	"code.google.com/p/mx3/data"
	"github.com/barnex/cuda5/cu"
	"log"
	"sync"
	"unsafe"
)

var (
	buf_lock sync.Mutex
	buf_pool map[int][]unsafe.Pointer
)

func GetBuffer(nComp int, m *data.Mesh) *data.Slice {
	buf_lock.Lock()
	defer buf_lock.Unlock()

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
		ptrs[i] = memAlloc(int64(cu.SIZEOF_FLOAT32 * N))
	}
	return data.SliceFromPtrs(m, data.GPUMemory, ptrs)
}

func RecycleBuffer(s *data.Slice) {
	buf_lock.Lock()
	defer buf_lock.Unlock()

	N := s.Mesh().NCell()
	pool := buf_pool[N]
	for i := 0; i < s.NComp(); i++ {
		pool = append(pool, s.DevPtr(i))
	}
	buf_pool[N] = pool
}
