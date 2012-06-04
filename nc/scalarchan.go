package nc

import ()

// ScalarChan is like a chan float32, but with fan-out
// (replicate data over multiple output channels).
// It can only have one input side though.
type ScalarChan struct {
	fanout []chan float32
}

func MakeSclarChan() (c ScalarChan) {
	return
}

// Add a new fanout and return it.
// All fanouts should be created before using the channel.
func (v *ScalarChan) Fanout(buf int) ScalarRecv {
	v.fanout = append(v.fanout, make(chan float32, buf))
	return v.fanout[len(v.fanout)-1]
}

// Send operator.
func (v *ScalarChan) Send(data float32) {
	if len(v.fanout) == 0 {
		panic("ScalarChan.Send: no fanout")
	}
	for i := range v.fanout {
		v.fanout[i] <- data
	}
}

// Receive-only side of a ScalarChan.
type ScalarRecv <-chan float32

// Receive operator.
func (r ScalarRecv) ScalarRecv() float32 {
	return <-r
}
