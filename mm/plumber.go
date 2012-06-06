package mm

import (
	"fmt"
	"log"
	"os"
)

func Connect3(dst Box, dstChan *[3]<-chan []float32, src Box, srcChan *[3]chan<- []float32, name string) {

	//stack runner here

	for i := 0; i < 3; i++ {
		Connect(dst, &(*dstChan)[i], src, &(*srcChan)[i], name+[3]string{"X", "Y", "Z"}[i])
	}

}

type conn struct {
	srcPtr  *chan<- []float32
	srcName string
	dstPtr  []*<-chan []float32
	dstName []string
}

var connections = make(map[string]*conn)

func getConn(name string) *conn {
	if c, ok := connections[name]; ok {
		return c
	}
	connections[name] = new(conn)
	return connections[name]
}

func Connect(dstBox Box, dstChan *<-chan []float32, srcBox Box, srcChan *chan<- []float32, name string) {
	c := getConn(name)
	if c.srcPtr != nil {
		panic(name + " already connected")
	}

	c.srcPtr = srcChan
	c.srcName = boxname(srcBox)

	c.dstPtr = append(c.dstPtr, dstChan)
	c.dstName = append(c.dstName, boxname(dstBox))

}

func boxname(value interface{}) string {
	return fmt.Sprintf("%T%p", value, value)
}

func ConnectNow() {

	dot, err := os.OpenFile("plumber.dot", os.O_WRONLY|os.O_CREATE, 0666)
	if err != nil {
		log.Println(err) //far from fatal
	} else {
		defer dot.Close()
	}

	// also run boxes here? +log: started...
	for name, c := range connections {

		if len(c.dstPtr) == 1 {
			fmt.Fprintln(dot, c.srcName, "->", c.dstName[0], "[label=", name, "]")
			ch := make(chan []float32, DefaultBufSize())
			*(c.dstPtr[0]) = ch
			*(c.srcPtr) = ch
		}

	}
}

func DefaultBufSize() int {
	return N / warp
}

type Box interface{}

// http://www.smbc-comics.com/index.php?db=comics&id=1330#comic
