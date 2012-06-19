package mm

import (
	. "nimble-cube/nc"
)

type GpuDemagBox struct {
	M [3]<-chan GpuBlock   "m"
	H [3][]chan<- GpuBlock "H"
}

func (box *GpuDemagBox) Run() {
	for {

	}
}
