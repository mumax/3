package nc

// Plumber connects the input/output channels of boxes.
// The resulting graph is saved in plumber.dot.

// TODO: Plumber type with methods, one global plumber.
// just for readability.
// or plumber package? plumber.Connect()...

import (
	"log"
	"reflect"
)

var (
	boxes []Box
	//	chanPtr              = make(map[string][]*[]chan<- []float32)
	//	chan3Ptr             = make(map[string][]*[3][]chan<- []float32)
	//	chanF64Ptr           = make(map[string][]*[]chan<- float64)
	srcBoxFor  = make(map[string]Box)  //IDEA: store vector/scalar kind here too
	srcChanFor = make(map[string]Chan) //IDEA: store vector/scalar kind here too
)

type Chan interface{}

func Start() {
	AutoConnectAll()
	dot.Close()
	StartBoxes()
}

func StartBoxes() {
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
func Register(box Box) { //, structTag ...string) {
	boxes = append(boxes, box)
	dot.AddBox(boxname(box))

	// find and map all output channels
	val := reflect.ValueOf(box).Elem()
	typ := val.Type()
	for i := 0; i < typ.NumField(); i++ {
		// use the field's struct tag as channel name
		name := string(typ.Field(i).Tag)
		if name == "" {
			continue
		}
		chanPtr := val.Field(i).Addr().Interface()
		if IsOutputChan(chanPtr) {
			if prev, ok := srcBoxFor[name]; ok {
				panic(name + " provided by both " + boxname(prev) + " and " + boxname(box))
			}
			RegisterChannel(box, chanPtr, name)
		}
	}
}

func RegisterChannel(box Box, chanPtr Chan, name string) {
	Assert(IsOutputChan(chanPtr))
	srcBoxFor[name] = box
	srcChanFor[name] = chanPtr
	log.Println("[plumber]", boxname(box), "provides", name)
	RegisterComponentChannels(box, chanPtr, name)
}

func RegisterComponentChannels(box Box, chanPtr Chan, name string) {
	switch c := chanPtr.(type) {
	default:
		return // nothing to see here
	case *[3][]chan<- []float32:
		RegisterChannel(box, &c[X], name+".x")
		RegisterChannel(box, &c[Y], name+".y")
		RegisterChannel(box, &c[Z], name+".z")
	}
}

func AutoConnectAll() {
	// connect all input channels using output channels from Register()
	for _, box := range boxes {
		AutoConnect(box)
	}
}

// Check if v, a pointer, can be used as an input channel.
// E.g.:
// 	*<-chan []float32, *[3]<-chan []float32, *<-chan float64
func IsInputChan(ptr interface{}) bool {
	switch ptr.(type) {
	case *<-chan []float32, *[3]<-chan []float32, *<-chan float64:
		return true
	}
	return false
}

// pure reflection should do even better here

// Check if v, a pointer, can be used as an output channel.
// E.g.:
// 	*[]chan<- []float32, *[][3]chan<- []float32, *[]chan<- float64
func IsOutputChan(ptr interface{}) bool {
	switch ptr.(type) {
	case *[]chan<- []float32, *[3][]chan<- []float32, *[]chan<- float64:
		return true
	}
	return false
}

// Connect fields with equal struct tags
func AutoConnect(box Box) {
	val := reflect.ValueOf(box).Elem()
	typ := val.Type()
	for i := 0; i < typ.NumField(); i++ {
		// use the field's struct tag as channel name
		name := string(typ.Field(i).Tag)
		if name == "" {
			continue
		}
		fieldaddr := val.Field(i).Addr().Interface()
		if !IsInputChan(fieldaddr) {
			continue
		}
		srcbox := srcBoxFor[name]
		if srcbox == nil {
			log.Println("autoconnect:", boxname(box), name, "has no input")
		} else {
			log.Println("autoconnect:", boxname(box), "<-", name, "<-", boxname(srcbox))
			srcaddr := FieldByTag(reflect.ValueOf(srcbox).Elem(), name).Addr().Interface()
			ConnectChannels(fieldaddr, srcaddr)
			dot.Connect(boxname(box), boxname(srcbox), name, 2)
		}
	}
}

func ManualConnect(dstBox Box, dstChanPtr interface{}, srcBox Box, srcChanPtr interface{}, name string) {
	ConnectChannels(dstChanPtr, srcChanPtr)
	dot.Connect(boxname(dstBox), boxname(srcBox), name, 2)
}

func FieldByTag(v reflect.Value, tag string) (field reflect.Value) {
	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		if string(t.Field(i).Tag) == tag {
			return v.Field(i)
		}
	}
	Panic(v, "has no tag", tag)
	return
}

// Try to connect dst and src based on their type.
// In any case, dst and src should hold pointers to
// some type of channel or an array of channels.
func ConnectChannels(dst, src interface{}) {
	switch d := dst.(type) {
	default:
		Panic("plumber cannot handle destination", reflect.TypeOf(d))
	case *<-chan []float32:
		ConnectChanOfSlice(d, src)
	case *[3]<-chan []float32:
		Connect3ChanOfSlice(d, src)
	case *<-chan float64:
		ConnectChanOfFloat64(d, src)
	}
}

// Try to connect dst based on the type of src.
// In any case, src should hold a pointer to
// some type of channel or an array of channels.
func ConnectChanOfSlice(dst *<-chan []float32, src interface{}) {
	switch s := src.(type) {
	default:
		Panic("plumber cannot handle", reflect.TypeOf(s))
	case *[]chan<- []float32:
		ch := make(chan []float32, DefaultBufSize())
		*s = append(*s, ch)
		*dst = ch
	}
}

// Try to connect dst based on the type of src.
// In any case, src should hold a pointer to
// some type of channel or an array of channels.
func Connect3ChanOfSlice(dst *[3]<-chan []float32, src interface{}) {
	switch s := src.(type) {
	default:
		Panic("plumber cannot handle", reflect.TypeOf(s))
	case *[3][]chan<- []float32:
		for i := 0; i < 3; i++ {
			ConnectChanOfSlice(&(*dst)[i], &(*s)[i])
		}
	}
}

// Try to connect dst based on the type of src.
// In any case, src should hold a pointer to
// some type of channel or an array of channels.
func ConnectChanOfFloat64(dst *<-chan float64, src interface{}) {
	switch s := src.(type) {
	default:
		Panic("plumber cannot handle", reflect.TypeOf(s))
	case *[]chan<- float64:
		ch := make(chan float64, 1)
		*s = append(*s, ch)
		*dst = ch
	}
}

type Box interface{}

type Runner interface {
	Run()
}

// http://www.smbc-comics.com/index.php?db=comics&id=1330#comic
