package nc

// Plumber connects the input/output channels of boxes
// on a best-effort basis. I.e. channels various types
// are connected, inserting conversions if necessary.
// Not all combinations of channels types are supported.

import (
	"reflect"
	"strings"
)

// Type alias for documentation
type Chan interface{}

// Try to connect dst and src based on their type.
// In any case, dst and src should hold pointers to
// some type of channel or an array of channels.
func Connect(dst, src Chan) {
	switch d := dst.(type) {
	default:
		cannotHandleDst(dst)
	case *<-chan []float32:
		connectChanOfSlice(d, src)
	case *[3]<-chan []float32:
		connect3ChanOfSlice(d, src)
	case *<-chan float64:
		connectChanOfFloat64(d, src)
	}
	RegisterConn(dst, src)
}

func cannotHandleDst(v interface{}) {
	typstr := reflect.TypeOf(v).String()
	if isOutputChan(v) {
		panic("plumber: " + typstr + " is a source, cannot use it as destination")
	}
	cannotHandle(typstr)
}

func cannotHandleSrc(v interface{}) {
	typstr := reflect.TypeOf(v).String()
	if isInputChan(v) {
		panic("plumber: " + typstr + " is a destination, cannot use it as source")
	}
	cannotHandle(typstr)
}
func cannotHandle(typstr string) {
	msg := "plumber: cannot handle " + typstr + ": need pointer to known channel type"
	if !strings.HasPrefix(typstr, "*") {
		msg += ": need pointer"
	}
	panic(msg)
}

// Try to connect dst based on the type of src.
// In any case, src should hold a pointer to
// some type of channel or an array of channels.
func connectChanOfSlice(dst *<-chan []float32, src interface{}) {
	switch s := src.(type) {
	default:
		cannotHandleSrc(src)
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
		cannotHandleSrc(src)
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
		cannotHandleSrc(src)
	case *[]chan<- float64:
		ch := make(chan float64, 1)
		*s = append(*s, ch)
		*dst = ch
	}
}

// http://www.smbc-comics.com/index.php?db=comics&id=1330#comic
