package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	MaxAngle  *GetScalar
	SpinAngle sSetter
)

func init() {
	MaxAngle = NewGetScalar("MaxAngle", "rad", "Maximum angle between neighboring spins", GetMaxAngle)
	SpinAngle.init("spinAngle", "rad", "Angle between neighboring spins", SetSpinAngle)
}

func SetSpinAngle(dst *data.Slice) {
	cuda.SetMaxAngle(dst, M.Buffer(), lex2.Gpu(), regions.Gpu(), M.Mesh())
}

func GetMaxAngle() float64 {
	s, recycle := SpinAngle.Slice()
	if recycle {
		defer cuda.Recycle(s)
	}
	return float64(cuda.MaxAbs(s)) // just a max would be fine, but not currently implemented
}
