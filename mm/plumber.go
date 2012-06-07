package mm

import ()

func Connect3(dst Box, dstFanout *[3]<-chan []float32, src Box, srcChan *[3][]chan<- []float32, name string) {
	dot.Connect(boxname(dst), boxname(src), name, 3)
	for i := 0; i < 3; i++ {
		connect(&(*dstFanout)[i], &(*srcChan)[i])
	}
}

func Connect(dst Box, dstChan *<-chan []float32, src Box, srcChan *[]chan<- []float32, name string) {
	dot.Connect(boxname(dst), boxname(src), name, 1)
	connect(dstChan, srcChan)
}

func connect(dst *<-chan []float32, src *[]chan<- []float32) {
	ch := make(chan []float32, DefaultBufSize()) // TODO: revise buffer size?
	*src = append(*src, ch)
	*dst = ch
}

func ConnectFloat64(dst Box, dstChan *<-chan float64, src Box, srcChan *[]chan<- float64, name string) {
	dot.Connect(boxname(dst), boxname(src), name, 1)
	ch := make(chan float64, 1)
	*srcChan = append(*srcChan, ch)
	*dstChan = ch
}

func DefaultBufSize() int {
	return N / warp
}

type Box interface{}

// http://www.smbc-comics.com/index.php?db=comics&id=1330#comic
