package mm

import (
	. "nimble-cube/nc"
)

// Send constant until time stops ticking.
func SendConst(tick <-chan float32, output Chan, value float32) {
	slice := make([]float32, warp)
	Memset(slice, value)

	for _ = range tick {
		output.Send(slice)
	}
}
