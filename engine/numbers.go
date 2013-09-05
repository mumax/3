package engine

// simple implementation of getVec
type GetFunc struct {
	doc
	get func() []float64
}

func newGetfunc(nComp int, name, unit, doc_ string, get func() []float64) *GetFunc {
	g := &GetFunc{doc{nComp, name, unit}, get}
	DeclROnly(name, g, doc_)
	return g
}

// INTERNAL
func NewGetScalar(name, unit, doc string, get func() float64) *GetFunc {
	return newGetfunc(1, name, unit, doc, func() []float64 {
		return []float64{get()}
	})
}

// INTERNAL
func NewGetVector(name, unit, doc string, get func() []float64) *GetFunc {
	return newGetfunc(3, name, unit, doc, get)
}

func (g *GetFunc) GetVec() []float64 {
	return g.get()
}
