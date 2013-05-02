package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

type setter struct {
	nComp int
	mesh  *data.Mesh
	autosave
	setFn func(dst *data.Slice, good bool) // calculates quantity and stores in dst
}

func newSetter(nComp int, m *data.Mesh, name, unit string, setFunc func(dst *data.Slice, good bool)) *setter {
	return &setter{nComp, m, autosave{name: name}, setFunc}
}

// notify the handle that it may need to be saved
func (b *setter) set(dst *data.Slice, goodstep bool) {
	b.setFn(dst, goodstep)
	if goodstep && b.needSave() {
		goSaveCopy(b.autoFname(), dst, Time)
		b.saved()
	}
}

// TODO: rm: use set + recylce
func (b *setter) get_mustRecyle() *data.Slice {
	buffer := cuda.GetBuffer(b.nComp, b.mesh)
	b.setFn(buffer, false)
	return buffer
}
