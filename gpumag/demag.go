package gpumag

import (
	"nimble-cube/core"
	"nimble-cube/gpu"
	"nimble-cube/gpu/conv"
	"nimble-cube/mag"
)

type Demag struct {
	convolution conv.Symm2D
	b           gpu.Chan3
}

const DEFAULT_DEMAG_ACCURACY = 8

func NewDemag(tag string, m gpu.RChan3, accuracy ...int) *Demag {
	d := new(Demag)
	acc := DEFAULT_DEMAG_ACCURACY
	kernel := mag.BruteKernel(core.ZeroPad(m.Mesh()), acc)
	b := gpu.MakeChan3(tag, m.Unit(), m.Mesh()) // TODO: choose blocks
	d.convolution = *conv.NewSymm2D(m.Size(), kernel, m, b)
	return d
}

func(d*Demag)Run(){
	d.convolution.Run()
}
