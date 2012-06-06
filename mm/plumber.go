package mm

import (
	. "nimble-cube/nc"
)

func DefaultBufSize() int {
	return N / warp
}

type Box interface{}

func Connect3(dst Box, dstChan *[3]<-chan []float32, src Box, srcFan *FanIn3, name string) {
	for i := 0; i < 3; i++ {
		ch := make(chan []float32, DefaultBufSize())
		(*dstChan)[i] = ch
		(*srcFan)[i] = append((*srcFan)[i], ch)
	}
}

// http://www.smbc-comics.com/index.php?db=comics&id=1330#comic
