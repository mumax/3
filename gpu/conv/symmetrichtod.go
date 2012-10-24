package conv

import (
	"nimble-cube/core"
	"nimble-cube/gpu"
)

// conv.Symmetric wrapped with uploaders/downloaders
// to accept host input/output.
type SymmetricHtoD struct {
	hostin        core.RChan3
	devin, devout [3]gpu.Chan1
	hostout       core.Chan3
	convolution   *Symm2D
}

func (c *SymmetricHtoD) Run() {
	// TODO: racy! push to stack and have automatically popped off?
	go NewDownloader(make3RChan(c.devout), c.hostout).Run() // devout -> hostout
	go NewUploader(c.hostin, c.devin).Run()                 // hostin -> devin
	c.convolution.Run()                                     // devin -> devout
}

func NewSymmetricHtoD(m *core.Mesh, kernel [3][3][][][]float32, input core.RChan3, output core.Chan3) *SymmetricHtoD {
	size := m.Size()
	c := new(SymmetricHtoD)
	for i := 0; i < 3; i++ {
		c.devin[i] = gpu.MakeChan("convIn", input.Unit(), m) // TODO: blocks??
		c.devout[i] = gpu.MakeChan("convOut", input.Unit(), m)
	}
	c.convolution = NewSymm2D(size, kernel, make3RChan(c.devin), c.devout)
	c.hostin = input
	c.hostout = output
	return c
}

func make3RChan(c [3]gpu.Chan1) [3]gpu.RChan {
	return [3]gpu.RChan{c[0].MakeRChan(), c[1].MakeRChan(), c[2].MakeRChan()}
}
