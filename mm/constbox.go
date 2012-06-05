package mm

import (
	. "nimble-cube/nc"
)

type ConstBox struct {
	value  float32
	output FanIn
}

func NewConstBox(value float32) *ConstBox {
	box := new(ConstBox)
	box.value = value
	return box
}

func (box *ConstBox) Run() {

	data := make([]float32, warp) // no Buffer(): should not be GC'd
	Memset(data, box.value)

	for {
		box.output.Send(data)
	}
}
