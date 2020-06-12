package engine

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	Ext_TopologicalChargeFiniteLattice        = NewScalarValue("ext_topologicalchargefinitelattice", "", "2D topological charge", GetTopologicalChargeFiniteLattice)
	Ext_TopologicalChargeDensityFiniteLattice = NewScalarField("ext_topologicalchargedensityfinitelattice", "",
		"2D topological charge density", SetTopologicalChargeDensityFiniteLattice)
)

func SetTopologicalChargeDensityFiniteLattice(dst *data.Slice) {
	cuda.SetTopologicalChargeFiniteLattice(dst, M.Buffer(), M.Mesh())
}

func GetTopologicalChargeFiniteLattice() float64 {
	s := ValueOf(Ext_TopologicalChargeDensityFiniteLattice)
	defer cuda.Recycle(s)
  	N := Mesh().Size()
	return (0.25 / math.Pi / float64(N[Z])) * float64(cuda.Sum(s))
}
