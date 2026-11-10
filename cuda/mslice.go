package cuda

import (
	"github.com/mumax/3/data"
	"unsafe"
)

// Slice + scalar multiplier.
type MSlice struct {
	arr  *data.Slice
	mul  []float64
	view bool // arr is not owned (not from the buffer pool): Recycle leaves it alone
}

func ToMSlice(s *data.Slice) MSlice {
	return MSlice{
		arr: s,
		mul: ones(s.NComp()),
	}
}

func MakeMSlice(arr *data.Slice, mul []float64) MSlice {
	return MSlice{arr: arr, mul: mul}
}

// ViewMSlice wraps a persistent (non-pooled) slice; Recycle will not free it.
func ViewMSlice(s *data.Slice) MSlice {
	return MSlice{arr: s, mul: ones(s.NComp()), view: true}
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

func (m MSlice) SetMul(c int, mul float32) {
	m.mul[c] = float64(mul)
}

func (m MSlice) Recycle() {
	if m.arr != nil && !m.view {
		Recycle(m.arr)
		m.arr = nil
	}
}

var _ones = [4]float64{1, 1, 1, 1}

func ones(n int) []float64 {
	return _ones[:n]

}
