package engine

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	Ext_TopologicalCharge        *GetScalar
	Ext_TopologicalChargeDensity sSetter
)

func init() {
	Ext_TopologicalCharge = NewGetScalar("ext_topologicalcharge", "", "Topological charge", GetTopologicalCharge)
	Ext_TopologicalChargeDensity.init("ext_topologicalchargedensity", "1/m2", "Topological charge density", SetTopologicalChargeDensity)
}

func SetTopologicalChargeDensity(dst *data.Slice) {
	cuda.SetTopologicalCharge(dst, M.Buffer(), M.Mesh())
}

func GetTopologicalCharge() float64 {
	s, recycle := Ext_TopologicalChargeDensity.Slice()
	if recycle {
		defer cuda.Recycle(s)
	}
	c := Mesh().CellSize()
	N := Mesh().Size()
	return (0.25 * c[0] * c[1] / math.Pi / float64(N[2])) * float64(cuda.Sum(s))
}
