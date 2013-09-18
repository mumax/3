package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

// quantity that is not stored, but can output to (set) a buffer
type setter struct {
	set func(dst *data.Slice) // calculates quantity and stores in dst
	info
}

// initialize setter and declare as quantity (for script and gui)
func (q *setter) init(nComp int, m *data.Mesh, name, unit, doc string, setFunc func(dst *data.Slice)) {
	*q = setter{setFunc, Info(nComp, name, unit, m)}
	DeclROnly(name, q, doc)
}

// get the quantity, recycle will be true (q needs to be recycled)
func (b *setter) Get() (q *data.Slice, recycle bool) {
	buffer := cuda.Buffer(b.nComp, b.mesh)
	b.set(buffer)
	return buffer, true // must recycle
}

func (s *setter) Save()           { Save(s) } // TODO: rm
func (s *setter) SaveAs(f string) { SaveAs(s, f) }
