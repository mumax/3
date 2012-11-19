package uni

import (
//"code.google.com/p/nimble-cube/nimble"
)

// Buffer buffers the original channel to hold
// an entire data frame at once. If the original
// is already sufficiently buffered, it is simply
// returned.
//func Buffer(c nimble.ChanN) nimble.ChanN {
//	if c.BufLen() >= c.Mesh().NCell() {
//		return c
//	}
//
//	mem := c.MemType()
//	out := nimble.MakeChanN(c.NComp(), c.Tag(), c.Unit(), c.Mesh(), mem, 0)
//
//	switch mem{
//		case nimble.CPUMemory, nimble.UnifiedMemory:
//		case nimble.GPUMemory:
//		default: panic("bug")
//	}
//
//	nimble.StackFunc(func() {
//		for {
//
//		}
//	})
//}
