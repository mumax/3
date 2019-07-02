package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	phi   = NewScalarField("phi", "rad", "Azimuthal angle", SetPhi)
	theta = NewScalarField("theta", "rad", "Polar angle", SetTheta)
)

func SetPhi(dst *data.Slice) {
	cuda.SetPhi(dst, M.Buffer())
}

func SetTheta(dst *data.Slice) {
	cuda.SetTheta(dst, M.Buffer())
}
