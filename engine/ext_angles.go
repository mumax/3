package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	ext_rxyphitheta = NewVectorField("ext_rxyphitheta", "m[rxy] rad[phi] rad[theta]", "Magnitude of m in xy plane, azimuthal angle, polar angle", SetRxyPhiTheta)
)

func SetRxyPhiTheta(dst *data.Slice) {
	cuda.SetRxyPhiTheta(dst, M.Buffer())
}
