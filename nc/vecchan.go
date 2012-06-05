package nc

import ()

// Like a [3]chan[]float32, but with fan-out
// (replicate data over multiple output channels).
// It can only have one input side though.
type FanIn3 struct {
	fanout [][3]chan []float32
}

func MakeFanIn3() (v FanIn3) {
	return
}

// Add a new fanout and return it.
// All fanouts should be created before using the channel.
func (v *FanIn3) Fanout(buf int) FanOut3 {
	v.fanout = append(v.fanout, [3]chan []float32{
		make(chan []float32, buf),
		make(chan []float32, buf),
		make(chan []float32, buf)})

	ch := v.fanout[len(v.fanout)-1]
	return FanOut3([3]<-chan []float32{ch[X], ch[Y], ch[Z]})
}

// Send operator.
// TODO: select loop so we can send in any order?
func (v *FanIn3) Send(data [3][]float32) {
	if len(v.fanout) == 0 {
		panic("FanIn3.Send: no fanouts")
	}
	for i := range v.fanout {
		v.fanout[i][X] <- data[X]
		v.fanout[i][Y] <- data[Y]
		v.fanout[i][Z] <- data[Z]
	}
}
