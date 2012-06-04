package mm // TODO: nc

import (
	"log"
)

type Chan chan []float32

func MakeChan(buf int) Chan {
	return make(Chan, buf)
}

func (v Chan) Send(data []float32) {
	log.Println(v, ".Send", data)
	v <- data
	log.Println(v, ".Send", "OK")
}

func (v Chan) Recv() (data []float32) {
	log.Println(v, ".Recv", "waiting")
	data = <-v
	log.Println(v, ".Recv", data)
	return
}
