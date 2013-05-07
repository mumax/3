package script

import (
	"bytes"
	"fmt"
	"testing"
)

func TestParser(t *testing.T) {
	src := bytes.NewBuffer([]byte(testText))
	l := newLexer(src)
	for f := parseLine(l); f != nil; f = parseLine(l) {
		fmt.Println(f())
		fmt.Println("\n=====")
	}
}

const testText = `
	a
	1
	a() 
	a(1)
	a(b)
	a(1, 2) 
	a(1, b(c()), d)
	x y
	9 x
	x 9
	()
`
