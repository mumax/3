package nimble

//TODO: time quantum, output quantum

var Time Clock

type Clock struct {
	timeOut []chan float64
}

func (c *Clock) Send(t float64) {
	for i := range c.timeOut {
		c.timeOut[i] <- t
	}
}

func (c *Clock) NewReader() <-chan float64 {
	ch := make(chan float64, 1)
	c.timeOut = append(c.timeOut, ch)
	return ch
}
