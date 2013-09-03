package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

// quantity that is not explicitly stored,
// but only added to an other quantity (like effective field contributions)
type adderQuant struct {
	addTo func(dst *data.Slice) // calculates quantity and add result to dst
	info
}

// constructor
func adder(nComp int, m *data.Mesh, name, unit string, addFunc func(dst *data.Slice)) adderQuant {
	return adderQuant{addFunc, info{nComp, name, unit, m}}
}

// Calcuates and returns the quantity. recycle is true:
// slice needs to be recycled.
func (a *adderQuant) GetGPU() (q *data.Slice, recycle bool) {
	buf := cuda.GetBuffer(a.NComp(), a.Mesh())
	cuda.Zero(buf)
	a.addTo(buf)
	return buf, true
}

func (a *adderQuant) Get() (q *data.Slice, recycle bool) { return a.GetGPU() }

//func (p *adderQuant) Save()                              { save(p) }
//func (p *adderQuant) SaveAs(fname string)                { saveAs(p, fname) }
