package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

type maskQuant struct {
	bufferedQuant
}

func newMask(nComp int, m *data.Mesh, name, unit string) *maskQuant {
	slice := data.NilSlice(nComp, m)
	return &maskQuant{buffered(slice, name, unit)}
}

// Set the value of all cell faces with their normal along direction. E.g.:
// 	SetAll(X, 1) // sets all faces in YZ plane to value 1.
func (m *maskQuant) SetAll(component int, value float64) {
	m.init()
	cuda.Memset(m.buffer.Comp(util.SwapIndex(component, 3)), float32(value))
}

func (m *maskQuant) Set(src *data.Slice) {
	m.init()
	m.bufferedQuant.Set(src)
}

// TODO: alloc one component at a time
func (m *maskQuant) init() {
	checkMesh() //engine
	if m.isNil() {
		m.buffer = cuda.NewSlice(3, m.mesh) // could alloc only needed components...
		cuda.Memset(m.buffer, 1, 1, 1)      // default value: all ones.
		onFree(func() { m.buffer.Free(); m.buffer = nil })
	}
}

func (m *maskQuant) isNil() bool {
	return m.buffer.DevPtr(0) == nil
}

func (m *maskQuant) Download() *data.Slice {
	if m.isNil() {
		s := data.NewSlice(m.NComp(), m.mesh)
		return s // TODO: memset 1s?
	} else {
		return m.buffer.HostCopy()
	}
}
