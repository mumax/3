package mm

import (
	. "nimble-cube/nc"
)

type ConstBox struct {
	value  float32
	Output []chan<- Block
	Time   <-chan float64 "time"
}

func NewConstBox(value float32) *ConstBox {
	box := new(ConstBox)
	box.value = value
	return box
}

func (box *ConstBox) Run() {

	data := MakeBlock(WarpSize()) // no Buffer(): should not be GC'd
	Memset(data.Contiguous(), box.value)

	for {
		RecvFloat64(box.Time)
		for s := 0; s < NumWarp(); s++ {
			Send(box.Output, data)
		}
	}
}
