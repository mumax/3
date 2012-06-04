package nc

import ()

// Chan is like a chan[]float32, but with fan-out
// (replicate data over multiple output channels).
// It can only have one input side though.
type Chan struct {
	fanout []chan []float32
}

func MakeChan() (c Chan) {
	return
}

// Add a new fanout and return it.
// All fanouts should be created before using the channel.
func (v *Chan) Fanout(buf int) Recv {
	v.fanout = append(v.fanout, make(chan []float32, buf))
	return v.fanout[len(v.fanout)-1]
}

// Send operator.
func (v *Chan) Send(data []float32) {
	if len(v.fanout) == 0 {
		panic("Chan.Send: no fanout")
	}
	for i := range v.fanout {
		v.fanout[i] <- data
	}
}

// Receive-only side of a Chan.
type Recv <-chan []float32

// Receive operator.
func (r Recv) Recv() []float32 {
	return <-r
}
