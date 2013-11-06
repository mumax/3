package engine

// unifies getScalar and getVector
type getFunc struct {
	doc
	get func() []float64
}

func (g *getFunc) TableData() []float64 { return g.get() }

func newGetfunc_(nComp int, name, unit, doc_ string, get func() []float64) getFunc {
	return getFunc{Doc(nComp, name, unit), get}
}

type GetScalar struct{ getFunc }
type GetVector struct{ getFunc }

func (g *GetScalar) Get() float64    { return g.get()[0] }
func (g *GetVector) Get() [3]float64 { return unslice(g.get()) }

// INTERNAL
func NewGetScalar(name, unit, doc string, get func() float64) *GetScalar {
	g := &GetScalar{newGetfunc_(1, name, unit, doc, func() []float64 {
		return []float64{get()}
	})}
	DeclROnly(name, g, cat(doc, unit))
	return g
}

// INTERNAL
func NewGetVector(name, unit, doc string, get func() []float64) *GetVector {
	g := &GetVector{newGetfunc_(3, name, unit, doc, get)}
	DeclROnly(name, g, cat(doc, unit))
	return g
}
