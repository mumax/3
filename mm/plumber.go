package mm

import ()

func Connect3(dst Box, dstFanout *[3]<-chan []float32, src Box, srcChan *[3][]chan<- []float32, name string) {
	for i := 0; i < 3; i++ {
		connect(&(*dstFanout)[i], &(*srcChan)[i])
	}
}

func Connect(dstBox Box, dstChan *<-chan []float32, srcBox Box, srcChan *[]chan<- []float32, name string) {
	connect(dstChan, srcChan)
}

func connect(dst *<-chan []float32, src *[]chan<- []float32) {
	ch := make(chan []float32, DefaultBufSize()) // TODO: revise buffer size?
	*src = append(*src, ch)
	*dst = ch
}

func DefaultBufSize() int {
	return N / warp
}

type Box interface{}

// http://www.smbc-comics.com/index.php?db=comics&id=1330#comic
