package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

// quantity that is not explicitly stored,
// but only added to an other quantity (like effective field)
type adder struct {
	autosave
	addFn func(dst *data.Slice) // calculates quantity and add result to dst
}

func newAdder(nComp int, m *data.Mesh, name, unit string, addFunc func(dst *data.Slice)) *adder {
	return &adder{newAutosave(nComp, name, unit, m), addFunc}
}

// Calls the addFunc to add the quantity to Dst. If output is needed,
// it is first added to a separate buffer, saved, and then added to Dst.
func (a *adder) addTo(dst *data.Slice, goodstep bool) {
	if goodstep && a.needSave() {
		buf := cuda.GetBuffer(dst.NComp(), dst.Mesh()) // TODO: not 3
		cuda.Zero(buf)
		a.addFn(buf)
		cuda.Madd2(dst, dst, buf, 1, 1)
		goSaveAndRecycle(a.autoFname(), buf, Time)
		a.saved()
	} else {
		a.addFn(dst)
	}
}

func (a *adder) Download() *data.Slice {
	b := cuda.GetBuffer(a.nComp, a.mesh)
	defer cuda.RecycleBuffer(b)
	cuda.Zero(b)
	a.addFn(b)
	return b.HostCopy()
}
