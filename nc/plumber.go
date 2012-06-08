package nc

// Plumber connects the input/output channels of boxes.
// The resulting graph is saved in plumber.dot.

import (
	"log"
	"reflect"
)

var boxes []Box

func Start() {
	connectBoxes()
	startBoxes()
}

func startBoxes() {
	for _, b := range boxes {
		if r, ok := b.(Runner); ok {
			log.Println("[plumber] starting:", boxname(r))
			go r.Run()
		} else {
			log.Println("[plumber] NOT STARTING:", boxname(b))
		}
	}
}

func Register(box Box) {
	boxes = append(boxes, box)
	log.Println("[plumber] register:", boxname(box))
}

func connectBoxes() {

	chanPtr := make(map[string][]*[]chan<- []float32)
	chan3Ptr := make(map[string][]*[3][]chan<- []float32)
	chanF64Ptr := make(map[string][]*[]chan<- float64)
	sourceBoxForChanName := make(map[string]string)

	// 1) find all output channels
	for _, box := range boxes {
		val := reflect.ValueOf(box).Elem()
		typ := val.Type()
		for i := 0; i < typ.NumField(); i++ {
			// use the field's struct tag as channel name
			name := string(typ.Field(i).Tag)
			if name == "" {
				continue
			}
			sourceBoxForChanName[name] = boxname(box)
			// store pointer to output channel
			fieldaddr := val.Field(i).Addr().Interface()
			switch ptr := fieldaddr.(type) {
			default:
				delete(sourceBoxForChanName, name)
				continue
			case *[]chan<- []float32:
				chanPtr[name] = append(chanPtr[name], ptr)
			case *[3][]chan<- []float32:
				chan3Ptr[name] = append(chan3Ptr[name], ptr)
			case *[]chan<- float64:
				chanF64Ptr[name] = append(chanF64Ptr[name], ptr)
			}
			log.Println("[plumber]", boxname(box), "->", name)
		}
	}

	//log.Println("[plumber] output fields:", chanPtr, chan3Ptr, chanF64Ptr)

	// 2) connect all input channels
	for _, box := range boxes {
		val := reflect.ValueOf(box).Elem()
		typ := val.Type()
		for i := 0; i < typ.NumField(); i++ {
			// use the field's struct tag as channel name
			name := string(typ.Field(i).Tag)
			if name == "" {
				continue
			}
			sBoxName := sourceBoxForChanName[name]
			if sBoxName == "" {
				log.Println("[plumber] NO INPUT FOR", boxname(box), name)
				continue
			}
			fieldaddr := val.Field(i).Addr().Interface()
			switch ptr := fieldaddr.(type) {
			default:
				continue
			case *<-chan []float32:
				dot.Connect(sourceBoxForChanName[name], boxname(box), name, 1)
				connect(ptr, chanPtr[name][0]) // TODO: handle multiple/none
			case *[3]<-chan []float32:
				dot.Connect(sourceBoxForChanName[name], boxname(box), name, 3)
				connect3(ptr, chan3Ptr[name][0]) // TODO: handle multiple/none
			}
			log.Println("[plumber]", sBoxName, "->", name, "->", boxname(box))
		}
	}
}

// Connect slice channels.
func connect(dst *<-chan []float32, src *[]chan<- []float32) {
	ch := make(chan []float32, DefaultBufSize())
	*src = append(*src, ch)
	*dst = ch
}

// Connect vector slice channels.
func connect3(dstFanout *[3]<-chan []float32, srcChan *[3][]chan<- []float32) {
	for i := 0; i < 3; i++ {
		connect(&(*dstFanout)[i], &(*srcChan)[i])
	}
}

// Connect scalar slices.

//
//
//// Connect single numbers (not space-dependent arrays).
//func ConnectFloat64(dst Box, dstChan *<-chan float64, src Box, srcChan *[]chan<- float64, name string) {
//	dot.Connect(boxname(dst), boxname(src), name, 1)
//	ch := make(chan float64, 1)
//	*srcChan = append(*srcChan, ch)
//	*dstChan = ch
//	stackBox(dst)
//	stackBox(src)
//}
//
//// dst has multiple inputs. Table, e.g.
//// TODO: -> output
//func ConnectManyFloat64(dst Box, dstChan *[]<-chan float64, src Box, srcChan *[]chan<- float64, name string) {
//	dot.Connect(boxname(dst), boxname(src), name, 1)
//	ch := make(chan float64, 1)
//	*srcChan = append(*srcChan, ch)
//	*dstChan = append(*dstChan, ch)
//	stackBox(dst)
//	stackBox(src)
//}

type Box interface{}
type Runner interface {
	Run()
}

// http://www.smbc-comics.com/index.php?db=comics&id=1330#comic
