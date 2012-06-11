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
		dot.Connect(boxName[DST], boxName[SRC], tagOfChan[conn[SRC]], 1)
	}

}

// Register a connection to appear in WriteGraph()
func RegisterConn(dst, src Chan) {
	connections = append(connections, [2]Chan{dst, src})
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

		ptr := field.Addr().Interface()

		if isChan(ptr) {
			boxOfChan[ptr] = box
		}

		if isOutputChan(ptr) {
			tagOfChan[ptr] = tag
			log.Println("tag of chan " + channame(ptr) + " = " + tag)
		}

		//	if isOutputChan(chanPtr) {
		//		if prev, ok := srcBoxForQuant[name]; ok {
		//			panic(name + " provided by both " + boxname(prev) + " and " + boxname(box))
		//		}
		//		RegisterTag(box, chanPtr, name)
		//	}
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

// Register a quantity taken form channel, give it a name.
//func RegisterTag(box Box, chanPtr Chan, name string) { // rm box, use boxforchanptr, mv SendQuant
//	Assert(isOutputChan(chanPtr))
//	setBoxFor(name, box)
//	setChanFor(name, chanPtr)
//	log.Println("[plumber]", boxname(box), "provides", name)
//	registerComponentQuants(box, chanPtr, name)
//}

// Automatically register components of vector quantities as "quant.x", ...
//func registerComponentQuants(box Box, chanPtr Chan, name string) { // rm box
//	Assert(isOutputChan(chanPtr))
//	switch c := chanPtr.(type) {
//	default:
//		return // nothing to see here
//	case *[3][]chan<- []float32:
//		RegisterTag(box, &c[X], name+".x")
//		RegisterTag(box, &c[Y], name+".y")
//		RegisterTag(box, &c[Z], name+".z")
//	case *[3][]chan<- float64:
//		RegisterTag(box, &c[X], name+".x")
//		RegisterTag(box, &c[Y], name+".y")
//		RegisterTag(box, &c[Z], name+".z")
//	}
//}

//func srcBoxFor(quant string) Box {
//	if b, ok := srcBoxForQuant[quant]; ok {
//		return b
//	}
//	panic("no such quantity " + quant)
//}
//
//func srcChanFor(quant string) Chan {
//	if b, ok := srcChanForQuant[quant]; ok {
//		return b
//	}
//	panic("no such quantity " + quant)
//}
//
//func setBoxFor(quant string, box Box) {
//	if _, ok := srcBoxForQuant[quant]; ok {
//		panic("already defined " + quant)
//	}
//	srcBoxForQuant[quant] = box
//}
//
//func setChanFor(quant string, c Chan) {
//	if _, ok := srcChanForQuant[quant]; ok {
//		panic("already defined " + quant)
//	}
//	srcChanForQuant[quant] = c
//}
