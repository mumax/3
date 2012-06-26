package safe

import (
	"github.com/barnex/cuda4/cu"
)

func makeslice(len_ int, elemsize int) slice {
	ptr := cu.MemAlloc(int64(len_) * int64(elemsize))
	return slice{ptr, len_, len_}
}

type slice struct {
	ptr_ cu.DevicePtr // address offset of first element
	len_ int          // number of elements
	cap_ int
}

func (s *slice) Pointer() cu.DevicePtr { return s.ptr_ }
func (s *slice) Len() int              { return s.len_ }
func (s *slice) Cap() int              { return s.cap_ }
func (s *slice) Free() {
	s.ptr_.Free()
	s.len_ = 0
	s.cap_ = 0
}

func (s *slice) slice(start, stop int) slice {
	if start >= s.cap_ || start < 0 || stop >= s.cap_ || stop < 0 {
		panic("slice index out of bounds")
	}
	if start > stop {
		panic("inverted slice range")
	}
	return slice{cu.DevicePtr(uintptr(s.ptr_) + uintptr(start)), stop - start, s.cap_ - start}
}
