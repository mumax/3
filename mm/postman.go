package mm

import (
	. "nimble-cube/nc"
)

func SendFloat64(fanout []chan<- float64, value float64) {
	for _, ch := range fanout {
		ch <- value
	}
}

func Send3(vectorFanout [3][]chan<- []float32, value [3][]float32) {
	for comp := 0; comp < 3; comp++ {
		for _, ch := range vectorFanout[comp] {
			ch <- value[comp]
		}
	}
}

func Send(fanout []chan<- []float32, value []float32) {
	for _, ch := range fanout {
		ch <- value
	}
}

func Recv3(vectorChan [3]<-chan []float32) [3][]float32 {
	// TODO: select?
	return [3][]float32{<-vectorChan[X], <-vectorChan[Y], <-vectorChan[Z]}
}

// The postman always rings three times.
