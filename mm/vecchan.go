package mm // TODO: nc

import (
	"log"
)

const (
	X = 0
	Y = 1
	Z = 2
)

type VecChan [3]chan []float32

func MakeVecChan(buf int) VecChan {
	return VecChan{make(chan []float32, buf), make(chan []float32, buf), make(chan []float32, buf)}
}

func (v *VecChan) Send(data [3][]float32) {
	if Debug {
		log.Println(v, ".Send", data)
	}
	// TODO: select loop so we can send in any order?
	v[X] <- data[X]
	v[Y] <- data[Y]
	v[Z] <- data[Z]
	if Debug {
		log.Println(v, ".Send", "OK")
	}
}

func (v *VecChan) Recv() (data [3][]float32) {
	if Debug {
		log.Println(v, ".Recv", "waiting")
	}
	data[X] = <-v[X]
	data[Y] = <-v[Y]
	data[Z] = <-v[Z]
	if Debug {
		log.Println(v, ".Recv", "data")
	}
	return
}
