package conv

import (
	"nimble-cube/core"
	"nimble-cube/gpu"
)

// conv.Symmetric wrapped with uploaders/downloaders
// to accept host input/output.
type SymmetricHtoD struct {
	devin, devout gpu.Chan3
	hostin        core.RChan3
	hostout       core.Chan3
	convolution   *Symm2
}

func (c *SymmetricHtoD) Run() {
	go gpu.NewUploader(c.hostin[0], c.devin[0]).Run()
	go gpu.NewUploader(c.hostin[1], c.devin[1]).Run()
	go gpu.NewUploader(c.hostin[2], c.devin[2]).Run()

	go gpu.NewDownloader(c.devout[0].ReadOnly(), c.hostout[0]).Run()
	go gpu.NewDownloader(c.devout[1].ReadOnly(), c.hostout[1]).Run()
	go gpu.NewDownloader(c.devout[2].ReadOnly(), c.hostout[2]).Run()

	c.convolution.Run()
}

func NewSymmetricHtoD(size [3]int, kernel [3][3][][][]float32, input core.RChan3, output core.Chan3) *SymmetricHtoD {
	c := new(SymmetricHtoD)
	c.devin = gpu.MakeChan3(size)
	c.devout = gpu.MakeChan3(size)
	c.convolution = NewSymm2(size, kernel, c.devin.ReadOnly(), c.devout)
	c.hostin = input
	c.hostout = output
	return c
}
