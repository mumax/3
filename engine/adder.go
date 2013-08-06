package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

// quantity that is not explicitly stored,
// but only added to an other quantity (like effective field contributions)
type adderQuant struct {
	autosave
	addFn func(dst *data.Slice) // calculates quantity and add result to dst
}

// constructor
func adder(nComp int, m *data.Mesh, name, unit string, addFunc func(dst *data.Slice)) adderQuant {
	return adderQuant{newAutosave(nComp, name, unit, m), addFunc}
}

// Calls the addFunc to add the quantity to Dst. If output is needed,
// it is first added to a separate buffer, saved, and then added to Dst.
func (a *adderQuant) addTo(dst *data.Slice, cansave bool) {
	if cansave && a.needSave() {
		buf, recycle := a.GetGPU()
		util.Assert(recycle == true)
		cuda.Madd2(dst, dst, buf, 1, 1)
		goSaveAndRecycle(a.autoFname(), buf, Time)
		a.saved()
	} else {
		a.addFn(dst)
	}
}

// Calcuates and returns the quantity. recycle is true:
// slice needs to be recycled.
func (a *adderQuant) GetGPU() (q *data.Slice, recycle bool) {
	buf := cuda.GetBuffer(a.NComp(), a.Mesh())
	cuda.Zero(buf)
	a.addFn(buf)
	return buf, true
}

func (a *adderQuant) Get() (q *data.Slice, recycle bool) {
	return a.GetGPU()
}

func (p *param) Save() {
	save(p)
}

func (p *param) SaveAs(fname string) {
	saveAs(p, fname)
}
