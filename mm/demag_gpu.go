package mm

import (
	. "nimble-cube/nc"
)


type GpuDemagBox struct {
	M [3]<-chan GPUSlice   "m"
	H [3][]chan<- GPUslice "H"
}

func (box *GpuDemagBox) Run() {
	for {


	}
}

