package mm

import (
	. "nimble-cube/nc"
)

type GpuDemagBox struct {
	M [3]<-chan GpuFloats   "m"
	H [3][]chan<- GpuFloats "H"
}

func (box *GpuDemagBox) Run() {
	for {

	}
}
