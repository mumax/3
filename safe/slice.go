package safe

import (
	"fmt"
	"github.com/barnex/cuda4/cu"
	"unsafe"
)

// internal base func for all makeXXX() functions
func makeslice(len_ int, elemsize int) slice {
	bytes := int64(len_) * int64(elemsize)
	ptr := cu.MemAlloc(bytes)
	cu.MemsetD8(ptr, 0, bytes)
	cu.CtxSynchronize()
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

func (dst *slice) copyHtoD(src unsafe.Pointer, srclen int, elemsize int) {
	if srclen != dst.Len() {
		panic(fmt.Errorf("len mismatch: len(src)=%v (host), dst.Len()=%v (device)", srclen, dst.Len()))
	}
	cu.MemcpyHtoD(dst.Pointer(), src, int64(elemsize)*int64(srclen))
}

func (src *slice) copyDtoH(dst unsafe.Pointer, dstlen int, elemsize int) {
	if dstlen != src.Len() {
		panic(fmt.Errorf("len mismatch: src.Len()=%v (device), len(dst)=%v (host)", src.Len(), dstlen))
	}
	cu.MemcpyDtoH(dst, src.Pointer(), int64(elemsize)*int64(dstlen))
}

func (dst *slice) copyDtoD(src *slice, elemsize int) {
	if dst.Len() != src.Len() {
		panic(fmt.Errorf("len mismatch: src.Len()=%v (device), dst.Len()=%v", src.Len(), dst.Len()))
	}
	cu.MemcpyDtoD(dst.Pointer(), src.Pointer(), int64(elemsize)*int64(dst.Len()))
}
