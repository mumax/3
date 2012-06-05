package nc

import ()

// Like a chan[]float32, but with fan-out
// (replicate data over multiple output channels).
// It can only have one input side though.
type FanIn struct {
	fanout []chan []float32
}

func MakeFanIn() (c FanIn) {
	return
}

// Add a new fanout and return it.
// All fanouts should be created before using the channel.
func (v *FanIn) FanOut(buf int) FanOut {
	v.fanout = append(v.fanout, make(chan []float32, buf))
	return v.fanout[len(v.fanout)-1]
}

// Send operator.
func (v *FanIn) Send(data []float32) {
	if len(v.fanout) == 0 {
		panic("FanIn.Send: no fanout")
	}
	for i := range v.fanout {
		v.fanout[i] <- data
	}
}

func (f *FanIn) Close() {
	for i := range f.fanout {
		close(f.fanout[i])
	}
}
