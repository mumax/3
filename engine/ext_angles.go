package engine

import (
	"github.com/mumax/3/v3/cuda"
	"github.com/mumax/3/v3/data"
)

var (
	ext_phi   = NewScalarField("ext_phi", "rad", "Azimuthal angle", SetPhi)
	ext_theta = NewScalarField("ext_theta", "rad", "Polar angle", SetTheta)
)

func SetPhi(dst *data.Slice) {
	cuda.SetPhi(dst, M.Buffer())
}

func SetTheta(dst *data.Slice) {
	cuda.SetTheta(dst, M.Buffer())
}
