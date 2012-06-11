package nc

// Wizzard inspects boxes using reflection and tries
// to connect as many channels as possible.
// Only channels with the same struct tag are connected.

import (
	"log"
	"reflect"
)

type Runner interface {
	Run()
}

// Type alias for documentation
type Box interface{}

// Auto-connect all registered boxes
//func AutoConnectAll() {
//	// connect all input channels using output channels from Register()
//	for _, box := range boxes {
//		AutoConnect(box)
//	}
//}

// Connect fields with equal struct tags
//func AutoConnect(box Box) {
//	val := reflect.ValueOf(box).Elem()
//	typ := val.Type()
//	for i := 0; i < typ.NumField(); i++ {
//		// use the field's struct tag as channel name
//		name := string(typ.Field(i).Tag)
//		if name == "" {
//			continue
//		}
//		fieldaddr := val.Field(i).Addr().Interface()
//		if !isInputChan(fieldaddr) {
//			continue
//		}
//		srcbox := srcBoxFor(name)
//		if srcbox == nil {
//			log.Println("autoconnect:", boxname(box), name, "has no input")
//		} else {
//			log.Println("autoconnect:", boxname(box), "<-", name, "<-", boxname(srcbox))
//			srcaddr := fieldByTag(reflect.ValueOf(srcbox).Elem(), name).Addr().Interface()
//			Connect(fieldaddr, srcaddr)
//			dot.Connect(boxname(box), boxname(srcbox), name, 2)
//		}
//	}
//}

func Start() {
	//AutoConnectAll()
	StartBoxes()
}

func StartBoxes() {
	for _, b := range boxes {
		if r, ok := b.(Runner); ok {
			log.Println("starting:", boxname(r))
			go r.Run()
		} else {
			log.Println("not starting:", boxname(b), ": no interface Run()")
		}
	}
}

// Retrieve a field by struct tag (instead of name).
func fieldByTag(v reflect.Value, tag string) (field reflect.Value) {
	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		if string(t.Field(i).Tag) == tag {
			return v.Field(i)
		}
	}
	Panic(v, "has no tag", tag)
	return
}

// Check if v, a pointer, can be used as an input channel.
// E.g.:
// 	*<-chan []float32, *[3]<-chan []float32, *<-chan float64
func isInputChan(ptr interface{}) bool {
	switch ptr.(type) {
	case *<-chan []float32, *[3]<-chan []float32, *<-chan float64:
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
