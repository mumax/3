package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"path"
)

// function that sets ("updates") quantity stored in dst
type updFunc func(dst *data.Slice)

// Output Handle for a quantity that is stored on the GPU.
type setter struct {
	nComp int
	mesh  *data.Mesh
	setFn func(dst *data.Slice) // calculates quantity and stores in dst
	autosave
}

func newSetter(nComp int, m *data.Mesh, name string, setFunc func(dst *data.Slice)) *setter {
	return &setter{nComp, m, f, autosave{name: name}}
}

func (b *setter) get_mustRecyle() *data.Slice {
	buffer := cuda.GetBuffer(b.nComp, b.mesh)
	b.setFn(buffer)
	return buffer
}

// notify the handle that it may need to be saved
func (b *setter) set(dst *data.Slice) {
	b.setFn(dst)
	if goodstep && b.needSave() {
		goSaveCopy(b.fname(), dst, Time)
		b.saved()
	}
}
