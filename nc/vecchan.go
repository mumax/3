package nc

import ()

// Chan is like a [3]chan[]float32, but with fan-out
// (replicate data over multiple output channels).
// It can only have one input side though.
type VecChan struct {
	fanout [][3]chan []float32
}

func MakeVecChan() (v VecChan) {
	return
}

// Add a new fanout and return it.
// All fanouts should be created before using the channel.
func (v *VecChan) Fanout(buf int) VecRecv {
	v.fanout = append(v.fanout, [3]chan []float32{
		make(chan []float32, buf),
		make(chan []float32, buf),
		make(chan []float32, buf)})

	ch := v.fanout[len(v.fanout)-1]
	return VecRecv([3]<-chan []float32{ch[X], ch[Y], ch[Z]})
}

// Send operator.
// TODO: select loop so we can send in any order?
func (v *VecChan) Send(data [3][]float32) {
	if len(v.fanout) == 0 {
		panic("VecChan.Send: no fanouts")
	}
	for i := range v.fanout {
		v.fanout[i][X] <- data[X]
		v.fanout[i][Y] <- data[Y]
		v.fanout[i][Z] <- data[Z]
	}
}

// Receive-only endpoint of a VecChan.
type VecRecv [3]<-chan []float32

// Vector receive operator.
func (v *VecRecv) Recv() [3][]float32 {
	return [3][]float32{<-v[X], <-v[Y], <-v[Z]}
}
