package data

import (
	"unsafe"
)

// Slice + scalar multiplier.
type MSlice struct {
	arr *Slice
	mul []float64
}

func (m MSlice) Size() [3]int {
	return m.arr.Size()
}

func (m MSlice) Len() int {
	return m.arr.Len()
}

func (m MSlice) DevPtr(c int) unsafe.Pointer {
	return m.arr.DevPtr(c)
}

func (m MSlice) Mul(c int) float32 {
	return float32(m.mul[c])
}
