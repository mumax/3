package nc

// Bookkeeper allows and boxes and channels to be registered
// so that an informative graphviz graph can be made.

import (
	"fmt"
	"log"
	"reflect"
	"strings"
	"unicode"
)

var (
	boxes       []Box                   // registered boxes
	connections [][2]Chan               // registered connections (dst, src)
	boxOfChan   = make(map[Chan]Box)    // maps channel onto its parent box
	tagOfChan   = make(map[Chan]string) // maps source channel onto its struct tag, if any.
	chanOfTag   = make(map[string]Chan) // maps struct tag on source channel.
)

// Write graphviz output.
func WriteGraph() {
	var dot graphvizwriter
	dot.Init("plumbing.dot")
	defer dot.Close()

	for _, box := range boxes {
		dot.AddBox(boxname(box))
	}

	const DST = 0
	const SRC = 1
	for _, conn := range connections {
		// connect to the parent box if registered,
		// use channel name otherwise.
		var boxName [2]string
		for i := 0; i < 2; i++ {
			if box, ok := boxOfChan[conn[i]]; ok {
				boxName[i] = boxname(box)
			} else {
				boxName[i] = channame(conn[i])
			}
		}

		// weight of the arrow is number of components.
		w := 1
		v := reflect.ValueOf(conn[SRC]).Elem()
		if v.Kind() == reflect.Array {
			w = v.Len()
		}
		dot.Connect(boxName[DST], boxName[SRC], tagOfChan[conn[SRC]], w)
	}

}

// Register a connection to appear in WriteGraph()
func RegisterConn(dst, src Chan) {
	connections = append(connections, [2]Chan{dst, src})
	//log.Println("connect", boxname(boxOfChan[dst]), tagOfChan[dst], channame(dst),
	//"<-", boxname(boxOfChan[src]), tagOfChan[src], channame(src))
}

// Register boxes to appear in WriteGraph()
func Register(box ...Box) {
	for _, b := range box {
		registerBox(b)
	}
}

// Checks if ptr is the address of a supported channel type.
func isChan(ptr interface{}) bool {
	return isInputChan(ptr) || isOutputChan(ptr)
}

func registerBox(box Box) {
	// append to boxes if not yet done so
	for _, b := range boxes {
		if b == box {
			return
		}
	}
	boxes = append(boxes, box)

	// find and map all output channels
	val := reflect.ValueOf(box).Elem()
	typ := val.Type()
	for i := 0; i < typ.NumField(); i++ {
		name := typ.Field(i).Name
		tag := string(typ.Field(i).Tag)
		field := val.Field(i)

		// skip non-exported fields
		if !unicode.IsUpper(rune(name[0])) { // who uses unicode identifiers anyway?
			continue
		}

		// set box of chan
		ptr := field.Addr().Interface()
		if isChan(ptr) {
			boxOfChan[ptr] = box
			// also set for components
			if field.Kind() == reflect.Array {
				for i := 0; i < field.Len(); i++ {
					boxOfChan[field.Index(i).Addr().Interface()] = box
				}
			}
		}

		// set tag of chan, chan of tag
		if isOutputChan(ptr) && tag != "" {
			tagOfChan[ptr] = tag
			chanOfTag[tag] = ptr // TODO: check multiple-defined
			log.Println("tag of chan " + channame(ptr) + " = " + tag)
		}
	}
}

func boxname(value Box) string {
	typ := fmt.Sprintf("%T", value)
	clean := typ[strings.Index(typ, ".")+1:] // strip "*mm."
	if strings.HasSuffix(clean, "Box") {
		clean = clean[:len(clean)-len("Box")]
	}
	return clean
}

func channame(c Chan) string {
	return fmt.Sprintf("chan0x%x", int(reflect.ValueOf(c).Elem().UnsafeAddr()))
}
