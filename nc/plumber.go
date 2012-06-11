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
		cannotConnect(dst, src)
	case *<-chan []float32:
		connectChanOfSlice(d, src)
	case *[3]<-chan []float32:
		connect3ChanOfSlice(d, src)
	case *<-chan float64:
		connectChanOfFloat64(d, src)
	case *[3]<-chan float64:
		connect3ChanOfFloat64(d, src)
	}
	RegisterConn(dst, src)
}

func cannotConnect(dst, src Chan) {
	if !isChan(dst) {
		cannotHandleType(dst)
	}
	if !isChan(src) {
		cannotHandleType(src)
	}
	if isInputChan(src) {
		panic("connect: " + reflect.TypeOf(src).String() + " is a source, cannot use it as destination")
	}
	if isOutputChan(dst) {
		panic("connect: " + reflect.TypeOf(dst).String() + " is a destination, cannot use it as source")
	}
	panic("connect: cannot match dst " + reflect.TypeOf(dst).String() + " to src " + reflect.TypeOf(src).String())
}

func cannotHandleType(v interface{}) {
	typstr := reflect.TypeOf(v).String()
	msg := "connect: cannot handle " + typstr + ": need pointer to known channel type"
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
		cannotConnect(dst, src)
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
		cannotConnect(dst, src)
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
		cannotConnect(dst, src)
	case *[]chan<- float64:
		ch := make(chan float64, 1)
		*s = append(*s, ch)
		*dst = ch
	}
}

func connect3ChanOfFloat64(dst *[3]<-chan float64, src interface{}) {
	switch s := src.(type) {
	default:
		cannotConnect(dst, src)
	case *[3][]chan<- float64:
		for i := 0; i < 3; i++ {
			connectChanOfFloat64(&(*dst)[i], &(*s)[i])
		}
	}
}

// Check if v, a pointer, can be used as an input channel.
// E.g.:
// 	*<-chan []float32, *[3]<-chan []float32, *<-chan float64
func isInputChan(ptr interface{}) bool {
	switch ptr.(type) {
	case *<-chan []float32, *[3]<-chan []float32, *<-chan float64, *[3]<-chan float64:
		return true
	}
	return false
}

// Check if v, a pointer, can be used as an output channel.
// E.g.:
// 	*[]chan<- []float32, *[][3]chan<- []float32, *[]chan<- float64
func isOutputChan(ptr interface{}) bool {
	switch ptr.(type) {
	case *[]chan<- []float32, *[3][]chan<- []float32, *[]chan<- float64, *[3][]chan<- float64:
		return true
	}
	return false
}

// http://www.smbc-comics.com/index.php?db=comics&id=1330#comic
