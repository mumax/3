package conv

import (
	"github.com/barnex/cuda4/safe"
	//"nimble-cube/core"
)

// General convolution, not optimized for specific cases.
type General struct {
	hostData
	deviceData3
	fwPlan safe.FFT3DR2CPlan
	bwPlan safe.FFT3DC2RPlan
}

func NewGeneral(input_, output_ [3][][][]float32, kernel [3][3][][][]float32) *General {
	c := new(General)
	c.hostData.init(input_, output_, kernel)
	return c
}
