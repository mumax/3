package nc

// Vets for bad connections between boxes.

import (
	"reflect"
	"unicode"
)

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
				Error("vet: not connected:", boxname(box), typ.Field(i).Name, tag)
				ok = false
			}
		}
	}
	if !ok {
		Panic("vet error(s)")
	}
}
