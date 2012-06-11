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
func AutoConnect(boxes ...Box) {
	Register(boxes...)

	for box := range boxes {

		val := reflect.ValueOf(box).Elem()
		typ := val.Type()
		for i := 0; i < typ.NumField(); i++ {

			// only consider tagged fields
			tag := string(typ.Field(i).Tag)
			if tag == "" {
				continue
			}

			// only consider input channels
			dst := val.Field(i).Addr().Interface()
			if !isInputChan(dst) {
				continue
			}

			src := chanOfTag[tag]
			if src != nil {
				Connect(dst, src)
			} else {
				log.Println("autoconnect: no source for", boxname(box), tag, channame(dst))
			}

		}

	}
}

// Call go box.Run() on all boxes.
func GoRun(box ...Runner) {
	for _, b := range box {
		log.Println("starting: " + boxname(b))
		go b.Run()
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
