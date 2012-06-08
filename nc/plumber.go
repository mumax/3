package nc

// Plumber connects the input/output channels of boxes.
// The resulting graph is saved in plumber.dot.

import (
	"log"
)

var (
	runners []Runner
)

type Runner interface {
	Run()
}

func stackBox(box Box) {
	if runner, ok := box.(Runner); ok {
		found := false
		for _, r := range runners {
			if r == runner {
				found = true
				break
			}
		}
		if !found {
			runners = append(runners, runner)
			//log.Println("[plumber] stack for Running:", boxname(box))
		}
	}
}

func GoRunBoxes(){
	for _,r:=range runners{
		log.Println("[plumber] starting:", boxname(r))
		go r.Run()
	}
}

// Connect vector slices.
func Connect3(dst Box, dstFanout *[3]<-chan []float32, src Box, srcChan *[3][]chan<- []float32, name string) {
	dot.Connect(boxname(dst), boxname(src), name, 3)
	for i := 0; i < 3; i++ {
		connect(&(*dstFanout)[i], &(*srcChan)[i])
	}
	stackBox(dst)
	stackBox(src)
}

// Connect scalar slices.
func Connect(dst Box, dstChan *<-chan []float32, src Box, srcChan *[]chan<- []float32, name string) {
	dot.Connect(boxname(dst), boxname(src), name, 1)
	connect(dstChan, srcChan)
	stackBox(dst)
	stackBox(src)
}

func connect(dst *<-chan []float32, src *[]chan<- []float32) {
	ch := make(chan []float32, DefaultBufSize()) // TODO: revise buffer size?
	*src = append(*src, ch)
	*dst = ch
}

// Connect single numbers (not space-dependent arrays).
func ConnectFloat64(dst Box, dstChan *<-chan float64, src Box, srcChan *[]chan<- float64, name string) {
	dot.Connect(boxname(dst), boxname(src), name, 1)
	ch := make(chan float64, 1)
	*srcChan = append(*srcChan, ch)
	*dstChan = ch
	stackBox(dst)
	stackBox(src)
}

// dst has multiple inputs. Table, e.g.
// TODO: -> output
func ConnectManyFloat64(dst Box, dstChan *[]<-chan float64, src Box, srcChan *[]chan<- float64, name string) {
	dot.Connect(boxname(dst), boxname(src), name, 1)
	ch := make(chan float64, 1)
	*srcChan = append(*srcChan, ch)
	*dstChan = append(*dstChan, ch)
	stackBox(dst)
	stackBox(src)
}

type Box interface{}

// http://www.smbc-comics.com/index.php?db=comics&id=1330#comic
