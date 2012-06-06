package mm

import (
	. "nimble-cube/nc"
)

func Send3(vectorChan [3]chan<- []float32, value [3][]float32) {
	// TODO: select?
	for i := 0; i < 3; i++ {
		vectorChan[i] <- value[i]
	}
}

func Recv3(vectorChan [3]<-chan []float32) [3][]float32 {
	// TODO: select?
	return [3][]float32{<-vectorChan[X], <-vectorChan[Y], <-vectorChan[Z]}
}

// The postman always rings three times.
