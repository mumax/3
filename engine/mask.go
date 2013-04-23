package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

// Stores values defined on the faces in-between cells.
// E.g.: the exchange interaction between cells.
// Automatically initialized to all ones.
type StaggeredMask struct {
	mask *data.Slice
}

func (m *StaggeredMask) SetAll(direction int, value float64) {
	m.init()
	cuda.Memset(m.mask.Comp(swapIndex(direction, 3)), float32(value))
}

func (m *StaggeredMask) SetLeftOf(direction int, ix, iy, iz int, value float64) {
	m.init()
	cuda.SetCell(m.mask, swapIndex(direction, 3), iz, iy, ix, float32(value))
}

func (m *StaggeredMask) SetRightOf(direction int, ix, iy, iz int, value float64) {
	m.init()
	i := [3]int{iz, ix, iy}
	i[direction]++
	size := m.mask.Mesh().Size()
	if i[direction] == size[direction] {
		i[direction] = 0 // wrap around boundary
	}
	cuda.SetCell(m.mask, swapIndex(direction, 3), iz, iy, ix, float32(value))
}

func (m *StaggeredMask) init() {
	if m.mask == nil {
		m.mask = cuda.NewSlice(3, Mesh()) // could alloc only needed components...
		cuda.Memset(m.mask, 1, 1, 1)      // default value: all ones.
		OnFree(func() { m.mask.Free(); m.mask = nil })
	}
}
