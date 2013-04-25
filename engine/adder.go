package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

// function that adds a quantity to dst
type addFunc func(dst *data.Slice)

// Output Handle for a quantity that is not explicitly stored,
// but only added to an other quantity (like effective field)
type adder struct {
	nComp int
	mesh  *data.Mesh
	addFn addFunc // calculates quantity and add result to dst
	autosave
}

func newAdder(nComp int, m *data.Mesh, name string, f addFunc) *adder {
	a := new(adder)
	a.nComp = nComp
	a.mesh = m
	a.addFn = f
	a.name = name
	return a
}

// Calls the addFunc to add the quantity to Dst. If output is needed,
// it is first added to a separate buffer, saved, and then added to Dst.
func (a *adder) addTo(dst *data.Slice, goodstep bool) {
	if goodstep && a.needSave() {
		buf := cuda.GetBuffer(dst.NComp(), dst.Mesh()) // TODO: not 3
		cuda.Zero(buf)
		a.addFn(buf)
		cuda.Madd2(dst, dst, buf, 1, 1)
		goSaveAndRecycle(a.fname(), buf, Time)
		a.saved()
	} else {
		a.addFn(dst)
	}
}

// Evaluates addFn and returns the result in a buffer.
// The returned buffer must be recycled with cuda.RecycleBuffer
func (a *adder) get_mustRecycle() *data.Slice {
	buf := cuda.GetBuffer(a.nComp, a.mesh)
	cuda.Zero(buf)
	a.addFn(buf)
	return buf
}
