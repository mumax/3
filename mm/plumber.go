package mm

import ()

func Connect3(dst Box, dstFanout *[3]<-chan []float32, src Box, srcChan *[3][]chan<- []float32, name string) {
	for i := 0; i < 3; i++ {
		connect(&(*dstFanout)[i], &(*srcChan)[i])
	}
}

func Connect(dstBox Box, dstChan *<-chan []float32, srcBox Box, srcChan *chan<- []float32, name string) {
	connect(dstChan, srcChan)
}

func connect(dstFanout *[]<-chan []float32, srcChan *chan<- []float32) {
	ch := make(chan []float32, DefaultBufSize()) // TODO: revise buffer size?
	*dstFanout = append(*dstFanout, ch)
	*srcChan = ch
}

//func WriteDot(fname string) {

//   dot, err := os.OpenFile(fname, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
//   if err != nil {
//   	log.Println(err) //far from fatal
//   	return
//   } else {
//   	defer dot.Close()
//   }

//   fmt.Fprintln(dot, "digraph dot{")

//   for name, c := range connections {

//   	fmt.Fprintln(dot, c.srcName, `[shape="rect"];`)
//   	if len(c.dstPtr) == 1 {
//   		fmt.Fprintln(dot, c.dstName[0], `[shape="rect"];`)
//   		fmt.Fprintln(dot, c.srcName, "->", c.dstName[0], "[label=", name, `];`)
//   	}

//   }

//   fmt.Fprintln(dot, "}")

//   err = exec.Command("dot", "-O", "-Tpdf", fname).Run()
//   if err != nil {
//   	log.Println(err)
//   }

//}
//func boxname(value interface{}) string {
//	typ := fmt.Sprintf("%T", value)
//	return typ[strings.Index(typ, ".")+1:]
//}

func DefaultBufSize() int {
	return N / warp
}

type Box interface{}

// http://www.smbc-comics.com/index.php?db=comics&id=1330#comic
