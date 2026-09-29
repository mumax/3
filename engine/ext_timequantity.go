package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/script"
)

func init() {
	DeclROnly("timeQ_s", TimeQ_s, "Current simulation time as a scalar Quantity (s)")
	DeclROnly("timeQ_v", TimeQ_v, "Current simulation time as a vector Quantity (s)")
	DeclROnly("timeQ_s2", TimeQ_s2, "Current simulation time as a scalar Quantity (s)")
	DeclROnly("timeQ_v2", TimeQ_v2, "Current simulation time as a vector Quantity (s)")
	DeclFunc("ScalarFuncQ", ScalarFuncQ, "Wraps a scalar function of time (e.g. sin(t)) as a Quantity usable in field expressions")
	DeclFunc("VectorFuncQ", VectorFuncQ, "Wraps three scalar functions of time as a vector Quantity (e.g. VectorFuncQ(sin(t), 0, cos(t)))")
}

var TimeQ_s = NewScalarField("TimeQ", "s", "Simulation time", func(dst *data.Slice) {
	cuda.Memset(dst, float32(Time))
})

var TimeQ_v = NewVectorField("TimeQVec", "s", "Simulation time (vector)", func(dst *data.Slice) {
	t := float32(Time)
	for c := 0; c < 3; c++ {
		cuda.Memset(dst.Comp(c), t)
	}
})

// A second potential implementation:

type timeQuantity struct {
	nComp int
}

var TimeQ_s2 Quantity = &timeQuantity{1}
var TimeQ_v2 Quantity = &timeQuantity{3}

func (t *timeQuantity) NComp() int { return t.nComp }

func (t *timeQuantity) EvalTo(dst *data.Slice) {
	v := float32(Time)
	for c := 0; c < t.nComp; c++ {
		cuda.Memset(dst.Comp(c), v)
	}
}

// functions of time as Quantities:

type scalarFuncQuantity struct {
	f script.ScalarFunction
}

func (q *scalarFuncQuantity) NComp() int { return 1 }

func (q *scalarFuncQuantity) EvalTo(dst *data.Slice) {
	cuda.Memset(dst.Comp(0), float32(q.f.Float()))
}

func ScalarFuncQ(f script.ScalarFunction) Quantity {
	return &scalarFuncQuantity{f}
}

type vectorFuncQuantity struct {
	fx, fy, fz script.ScalarFunction
}

func (q *vectorFuncQuantity) NComp() int { return 3 }

func (q *vectorFuncQuantity) EvalTo(dst *data.Slice) {
	cuda.Memset(dst.Comp(0), float32(q.fx.Float()))
	cuda.Memset(dst.Comp(1), float32(q.fy.Float()))
	cuda.Memset(dst.Comp(2), float32(q.fz.Float()))
}

func VectorFuncQ(fx, fy, fz script.ScalarFunction) Quantity {
	return &vectorFuncQuantity{fx, fy, fz}
}
