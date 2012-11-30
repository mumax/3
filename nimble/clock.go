package nimble

//TODO: time quantum, output quantum

var Clock clock

type Time struct{
	Time, Dt float64
	Valid bool
}

type clock struct {
	timeOut []chan Time
}

func (c *clock) Send(t Time) {
	for i := range c.timeOut {
		c.timeOut[i] <- t
	}
}

func (c *clock) NewReader() <-chan Time {
	ch := make(chan Time, 1)
	c.timeOut = append(c.timeOut, ch)
	return ch
}
