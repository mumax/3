package safe

import (
	"github.com/barnex/cuda4/cu"
)

// internal base func for all makeXXX() functions
func makeslice(len_ int, elemsize int) slice {
	ptr := cu.MemAlloc(int64(len_) * int64(elemsize))
	return slice{ptr, len_, len_}
}

// internal base type for all slices
type slice struct {
	ptr_ cu.DevicePtr // address offset of first element
	len_ int          // number of elements
	cap_ int
}

// Pointer to the first element.
func (s *slice) Pointer() cu.DevicePtr { return s.ptr_ }

// Slice length (number of elements).
func (s *slice) Len() int { return s.len_ }

// Slice capacity.
func (s *slice) Cap() int { return s.cap_ }

// Free the underlying storage.
// To be used with care. Free() should only be called on
// a slice created by MakeXXX(), not on a slice created
// by x.Slice(). Freeing a slice invalidates all other
// slices referring to it.
func (s *slice) Free() {
	s.ptr_.Free()
	s.len_ = 0
	s.cap_ = 0
}

// internal base func for all slice() functions
func (s *slice) slice(start, stop int, elemsize uintptr) slice {
	if start >= s.cap_ || start < 0 || stop >= s.cap_ || stop < 0 {
		panic("slice index out of bounds")
	}
	if start > stop {
		panic("inverted slice range")
	}
	return slice{cu.DevicePtr(uintptr(s.ptr_) + uintptr(start)*elemsize), stop - start, s.cap_ - start}
}
