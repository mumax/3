package engine

//TODO: time quantum, output quantum

//var Clock clock
//
//type Time struct {
//	Time  float64
//	Stage bool
//}
//
//type clock struct {
//	timeOut []chan Time
//}
//
//func (c *clock) Send(time float64, stage bool) {
//	for i := range c.timeOut {
//		c.timeOut[i] <- Time{time, stage}
//	}
//}
//
//func (c *clock) NewReader() <-chan Time {
//	ch := make(chan Time, 1)
//	c.timeOut = append(c.timeOut, ch)
//	return ch
//}
