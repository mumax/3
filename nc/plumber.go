package nc

// Plumber connects the input/output channels of boxes
// on a best-effort basis. I.e. channels various types
// are connected, inserting conversions if necessary.
// But not all combinations of channels types are supported.

import (
	"log"
	"reflect"
)

func ManualConnect(dstBox Box, dstChanPtr interface{}, srcBox Box, srcChanPtr interface{}, name string) {
	ConnectChannels(dstChanPtr, srcChanPtr)
	dot.Connect(boxname(dstBox), boxname(srcBox), name, 2)
	log.Println("connect:", boxname(dstBox), "<-", name, "<-", boxname(srcBox))
}

func ConnectToTag(box Box, boxInput Chan, chanName string) { // rm box, mv RecvQuant
	ManualConnect(box, boxInput, srcBoxFor(chanName), srcChanFor(chanName), chanName)
}

// Try to connect dst and src based on their type.
// In any case, dst and src should hold pointers to
// some type of channel or an array of channels.
func ConnectChannels(dst, src interface{}) {
	switch d := dst.(type) {
	default:
		Panic("plumber cannot handle destination", reflect.TypeOf(d))
	case *<-chan []float32:
		connectChanOfSlice(d, src)
	case *[3]<-chan []float32:
		connect3ChanOfSlice(d, src)
	case *<-chan float64:
		connectChanOfFloat64(d, src)
	}
}

// Try to connect dst based on the type of src.
// In any case, src should hold a pointer to
// some type of channel or an array of channels.
func connectChanOfSlice(dst *<-chan []float32, src interface{}) {
	switch s := src.(type) {
	default:
		Panic("plumber cannot handle", reflect.TypeOf(s))
	case *[]chan<- []float32:
		ch := make(chan []float32, DefaultBufSize())
		*s = append(*s, ch)
		*dst = ch
	}
}

// Try to connect dst based on the type of src.
// In any case, src should hold a pointer to
// some type of channel or an array of channels.
func connect3ChanOfSlice(dst *[3]<-chan []float32, src interface{}) {
	switch s := src.(type) {
	default:
		Panic("plumber cannot handle", reflect.TypeOf(s))
	case *[3][]chan<- []float32:
		for i := 0; i < 3; i++ {
			connectChanOfSlice(&(*dst)[i], &(*s)[i])
		}
	}
}

// Try to connect dst based on the type of src.
// In any case, src should hold a pointer to
// some type of channel or an array of channels.
func connectChanOfFloat64(dst *<-chan float64, src interface{}) {
	switch s := src.(type) {
	default:
		Panic("plumber cannot handle", reflect.TypeOf(s))
	case *[]chan<- float64:
		ch := make(chan float64, 1)
		*s = append(*s, ch)
		*dst = ch
	}
}

// Type alias for documentation
type Box interface{}

// Type alias for documentation
type Chan interface{}

type Runner interface {
	Run()
}

// http://www.smbc-comics.com/index.php?db=comics&id=1330#comic
