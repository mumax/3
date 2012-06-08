package nc

// Plumber connects the input/output channels of boxes.
// The resulting graph is saved in plumber.dot.

import (
	"log"
	"reflect"
)

var (
	boxes                []Box
	chanPtr              = make(map[string][]*[]chan<- []float32)
	chan3Ptr             = make(map[string][]*[3][]chan<- []float32)
	chanF64Ptr           = make(map[string][]*[]chan<- float64)
	sourceBoxForChanName = make(map[string]string)
)

func Start() {
	connectBoxes()
	dot.Close()
	startBoxes()
}

func startBoxes() {
	for _, b := range boxes {
		if r, ok := b.(Runner); ok {
			log.Println("[plumber] starting:", boxname(r))
			go r.Run()
		} else {
			log.Println("[plumber] not starting:", boxname(b), ": needs input")
		}
	}
}

// 
// Ad-hoc struct tags may be provided to map 
// field names of generic boxes to channel names.
// E.g.: `Output:alpha,Input:Time`
func Register(box Box, structTag ...string) {
	if len(structTag) > 1 {
		panic("too many struct tags")
	}
	boxes = append(boxes, box)
	dot.AddBox(boxname(box))

	// find and map all output channels
	val := reflect.ValueOf(box).Elem()
	typ := val.Type()
	for i := 0; i < typ.NumField(); i++ {
		// use the field's struct tag as channel name
		name := string(typ.Field(i).Tag)
		if name == "" && len(structTag) > 0 {
			name = reflect.StructTag(structTag[0]).Get(typ.Field(i).Name)
			if name != "" {
				log.Println("[plumber]", boxname(box), typ.Field(i).Name, "used as", name)
			}
		}
		if name == "" {
			continue
		}
		// store pointer to output channel
		fieldaddr := val.Field(i).Addr().Interface()
		switch ptr := fieldaddr.(type) {
		default:
			continue
		case *[]chan<- []float32:
			chanPtr[name] = append(chanPtr[name], ptr)
		case *[3][]chan<- []float32:
			chan3Ptr[name] = append(chan3Ptr[name], ptr)
		case *[]chan<- float64:
			chanF64Ptr[name] = append(chanF64Ptr[name], ptr)
		}
		// found one = ok
		sourceBoxForChanName[name] = boxname(box)
		log.Println("[plumber]", boxname(box), "->", name)
	}
}

func connectBoxes() {
	// connect all input channels using output channels from Register()
	for _, box := range boxes {
		val := reflect.ValueOf(box).Elem()
		typ := val.Type()
		for i := 0; i < typ.NumField(); i++ {
			// use the field's struct tag as channel name
			name := string(typ.Field(i).Tag)
			if name == "" {
				continue
			}
			fieldaddr := val.Field(i).Addr().Interface()
			sBoxName := sourceBoxForChanName[name]
			switch ptr := fieldaddr.(type) {
			default:
				continue
			case *<-chan []float32:
				if chanPtr[name] == nil {
					log.Println("[plumber] no input for", boxname(box), name)
					break
				}
				connect(ptr, chanPtr[name][0]) // TODO: handle multiple/none
				dot.Connect(boxname(box), sourceBoxForChanName[name], name, 2)
			case *[3]<-chan []float32:
				if chan3Ptr[name] == nil {
					log.Println("[plumber] no input for", boxname(box), name)
					break
				}
				connect3(ptr, chan3Ptr[name][0]) // TODO: handle multiple/none
				dot.Connect(boxname(box), sourceBoxForChanName[name], name, 3)
			case *<-chan float64:
				if chanF64Ptr[name] == nil {
					log.Println("[plumber] no input for", boxname(box), name)
					break
				}
				connectF64(ptr, chanF64Ptr[name][0]) // TODO: handle multiple/none
				dot.Connect(boxname(box), sourceBoxForChanName[name], name, 1)
			}
			if sBoxName != "" {
				log.Println("[plumber]", sBoxName, "->", name, "->", boxname(box))
			}
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

// Connect float64 channels.
func connectF64(dst *<-chan float64, src *[]chan<- float64) {
	ch := make(chan float64, 1)
	*src = append(*src, ch)
	*dst = ch
}

type Box interface{}

type Runner interface {
	Run()
}

// http://www.smbc-comics.com/index.php?db=comics&id=1330#comic
