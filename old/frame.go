package nimble

import (
	"unsafe"
	//"github.com/barnex/cuda5/safe"
)

type Tensor struct {
	ptr   [MAX_COMP]unsafe.Pointer
	len_  int32
	nComp int8
	MemType
}

func (f *Tensor) NComp() int {
	return int(f.nComp)
}

func (f *Tensor) Comp(i int) *Tensor {
	t := &Tensor{[MAX_COMP]unsafe.Pointer{f.ptr[i]}, f.len_, 1, f.MemType}
	return t
}

func (f *Tensor) Ptr(i int) unsafe.Pointer {
	return f.ptr[i]
}

func (f *Tensor) Len() int {
	return int(f.len_)
}

func (t Tensor) test_esc() unsafe.Pointer {
	return t.Comp(0).Ptr(0)
}
