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
	a()
	a(1) 
	a(1, 2) 
	a(b(), c(d()))
	a(b)
	@
	"
`
