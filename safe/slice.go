package safe

// INTERNAL.
// This file implements common functionality for all slice types
// (Float32s, Float64s, Complex64s, ...).

import (
	"fmt"
	"github.com/barnex/cuda4/cu"
	"unsafe"
)

// internal base func for all makeXXX() functions
func makeslice(len_ int, elemsize int) slice {
	bytes := int64(len_) * int64(elemsize)
	s := slice{0, len_, len_}
	if bytes > 0 {
		s.ptr_ = cu.MemAlloc(bytes)
		cu.MemsetD8(s.ptr_, 0, bytes)
		cu.CtxSynchronize()
	}
	return s
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
	if start >= s.cap_ || start < 0 || stop > s.cap_ || stop < 0 {
		panic("cuda4/safe: slice index out of bounds")
	}
	if start > stop {
		panic("cuda4/safe: inverted slice range")
	}
	return slice{cu.DevicePtr(uintptr(s.ptr_) + uintptr(start)*elemsize), stop - start, s.cap_ - start}
}

func (dst *slice) copyHtoD(src unsafe.Pointer, srclen int, elemsize int) {
	if srclen != dst.Len() {
		panic(fmt.Errorf("cuda4/safe: len mismatch: len(src)=%v (host), dst.Len()=%v (device)", srclen, dst.Len()))
	}
	cu.MemcpyHtoD(dst.Pointer(), src, int64(elemsize)*int64(srclen))
}

func (src *slice) copyDtoH(dst unsafe.Pointer, dstlen int, elemsize int) {
	if dstlen != src.Len() {
		panic(fmt.Errorf("cuda4/safe: len mismatch: src.Len()=%v (device), len(dst)=%v (host)", src.Len(), dstlen))
	}
	cu.MemcpyDtoH(dst, src.Pointer(), int64(elemsize)*int64(dstlen))
}

func (dst *slice) copyDtoD(src *slice, elemsize int) {
	if dst.Len() != src.Len() {
		panic(fmt.Errorf("cuda4/safe: len mismatch: src.Len()=%v (device), dst.Len()=%v", src.Len(), dst.Len()))
	}
	cu.MemcpyDtoD(dst.Pointer(), src.Pointer(), int64(elemsize)*int64(dst.Len()))
}

func (dst *slice) copyHtoDAsync(src unsafe.Pointer, srclen int, elemsize int, stream cu.Stream) {
	if srclen != dst.Len() {
		panic(fmt.Errorf("cuda4/safe: len mismatch: len(src)=%v (host), dst.Len()=%v (device)", srclen, dst.Len()))
	}
	cu.MemcpyHtoDAsync(dst.Pointer(), src, int64(elemsize)*int64(srclen), stream)
}

func (src *slice) copyDtoHAsync(dst unsafe.Pointer, dstlen int, elemsize int, stream cu.Stream) {
	if dstlen != src.Len() {
		panic(fmt.Errorf("cuda4/safe: len mismatch: src.Len()=%v (device), len(dst)=%v (host)", src.Len(), dstlen))
	}
	cu.MemcpyDtoHAsync(dst, src.Pointer(), int64(elemsize)*int64(dstlen), stream)
}

func (dst *slice) copyDtoDAsync(src *slice, elemsize int, stream cu.Stream) {
	if dst.Len() != src.Len() {
		panic(fmt.Errorf("cuda4/safe: len mismatch: src.Len()=%v (device), dst.Len()=%v", src.Len(), dst.Len()))
	}
	cu.MemcpyDtoDAsync(dst.Pointer(), src.Pointer(), int64(elemsize)*int64(dst.Len()), stream)
}

// Manually set the pointer, length and capacity.
// Side-steps the security mechanisms, use with caution.
func (s *slice) UnsafeSet(pointer unsafe.Pointer, length, capacity int) {
	s.ptr_ = cu.DevicePtr(pointer)
	s.len_ = length
	s.cap_ = capacity
}
