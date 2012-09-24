package conv

import (
	"nimble-cube/core"
	"nimble-cube/gpu"
)

// conv.Symmetric wrapped with uploaders/downloaders
// to accept host input/output.
type SymmetricHtoD struct {
	hostin        core.RChan3
	devin, devout [3]gpu.Chan
	hostout       core.Chan3
	convolution   *Symm2
}

func (c *SymmetricHtoD) Run() {
	go NewUploader(c.hostin, c.devin).Run()                 // hostin -> devin
	go NewDownloader(make3RChan(c.devout), c.hostout).Run() // devout -> hostout
	c.convolution.Run()                                     // devin -> devout
}

func NewSymmetricHtoD(size [3]int, kernel [3][3][][][]float32, input core.RChan3, output core.Chan3) *SymmetricHtoD {
	c := new(SymmetricHtoD)
	panic("todo")
	for i := 0; i < 3; i++ {
		c.devin[i] = gpu.MakeChan(size)
		c.devout[i] = gpu.MakeChan(size)
	}
	c.convolution = NewSymm2(size, kernel, make3RChan(c.devin), c.devout)
	c.hostin = input
	c.hostout = output
	return c
}

func make3RChan(c [3]gpu.Chan) [3]gpu.RChan {
	return [3]gpu.RChan{c[0].MakeRChan(), c[1].MakeRChan(), c[2].MakeRChan()}
}
