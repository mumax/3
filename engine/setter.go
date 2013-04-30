package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

// Output Handle for a quantity that is stored on the GPU.
type setter struct {
	nComp int
	mesh  *data.Mesh
	setFn func(dst *data.Slice, good bool) // calculates quantity and stores in dst
	autosave
}

func newSetter(nComp int, m *data.Mesh, name string, setFunc func(dst *data.Slice, good bool)) *setter {
	return &setter{nComp, m, setFunc, autosave{name: name}}
}

func (b *setter) get_mustRecyle() *data.Slice {
	buffer := cuda.GetBuffer(b.nComp, b.mesh)
	b.setFn(buffer, false)
	return buffer
}

// notify the handle that it may need to be saved
func (b *setter) set(dst *data.Slice, goodstep bool) {
	b.setFn(dst, goodstep)
	if goodstep && b.needSave() {
		goSaveCopy(b.fname(), dst, Time)
		b.saved()
	}
}
