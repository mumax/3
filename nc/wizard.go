package nc

// Wizzard inspects boxes using reflection and tries
// to connect as many channels as possible.
// Only channels with the same struct tag are connected.

import (
	"log"
	"reflect"
	"unicode"
)

// Runner is usually a Box whose Run() will loop forever
// processing Chan input.
type Runner interface {
	Run()
}

// Box represents a struct with input and output channels
// used to send data. Boxes usually have a Run() method
// that will loop forever reading input and writing output.
type Box interface{}

// Try to connect the boxes on a best-effort basis.
// Will connect fields with equal struct tags (e.g. "m") 
// that have not yet been connected.
// For lazy people.
func AutoConnect(boxes ...Box) {
	Register(boxes...)

	for _, box := range boxes {
		val := reflect.ValueOf(box).Elem()
		typ := val.Type()

		for i := 0; i < typ.NumField(); i++ {
			field := val.Field(i)

			// skip unexported
			if unicode.IsLower(rune(typ.Field(i).Name[0])) {
				log.Println("autoconnect: skipping", boxname(box), typ.Field(i).Name, ": unexported")
				continue
			}

			// only consider input channels
			dst := field.Addr().Interface()
			if !isInputChan(dst) {
				continue
			}

			// skip untagged fields
			tag := string(typ.Field(i).Tag)
			if tag == "" {
				log.Println("autoconnect: skipping", boxname(box), typ.Field(i).Name, ": no struct tag")
				continue
			}

			// skip already connected destinations
			if isConnected(field) {
				log.Println("autoconnect: skipping", boxname(box), tag, ": already connected")
				continue
			}

			// now the easy part: actually connect.
			src := chanOfTag[tag]
			if src != nil {
				log.Println("autoconnect:", boxname(box), tag, "<-", channame(src))
				Connect(dst, src)
			} else {
				log.Println("autoconnect: no source for", boxname(box), tag, channame(dst))
			}
		}
	}
}

func isConnected(field reflect.Value) bool {
	switch {
	default:
		Panic("isconnected: unexpected kind:", field.Kind())
	case field.Kind() == reflect.Array:
		return !field.Index(0).IsNil() // [3]chan is connected if elem 0 is connected
	case field.Kind() == reflect.Chan:
		return !field.IsNil() // chan is connected if not nil
	case field.Kind() == reflect.Slice: // Output fanout
		return !(field.Len() == 0)
	}
	panic(0)
	return false // silence 6g
}

// Check boxes for unconnected inputs.
// TODO: sources connected to the same dest box twice might also need checking
func Vet(boxes ...Box) {
	Register(boxes...)
	ok := true
	for _, box := range boxes {
		val := reflect.ValueOf(box).Elem()
		typ := val.Type()

		for i := 0; i < typ.NumField(); i++ {
			field := val.Field(i)

			// skip unexported
			if unicode.IsLower(rune(typ.Field(i).Name[0])) {
				continue
			}

			// only consider channels
			dst := field.Addr().Interface()
			if !isChan(dst) {
				continue
			}

			// check connected
			tag := string(typ.Field(i).Tag)
			if !isConnected(field) {
				log.Println("vet: not connected:", boxname(box), typ.Field(i).Name, tag)
				ok = false
			}
		}
	}
	if !ok {
		Panic("vet error(s)")
	}
}

// Vet and Run all boxes.
func GoRun(box ...Runner) {
	// first vet all boxes at once
	var boxes []Box
	for _, b := range box {
		boxes = append(boxes, Box(b))
	}
	Vet(boxes...)

	// only then run them
	for _, b := range box {
		log.Println("starting: " + boxname(b))
		go b.Run()
	}
}

// Run all boxes that have been registered or autoconnected.
// For lazy people.
func AutoRun() {
	for _, b := range boxes {
		if r, ok := b.(Runner); ok {
			GoRun(r)
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
