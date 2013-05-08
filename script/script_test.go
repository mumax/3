package script

import (
	"bytes"
	"fmt"
	"io"
	"testing"
)

func TestParser(t *testing.T) {
	src := bytes.NewBuffer([]byte(testText))
	l := newLexer(src)
	expr, err := parseLine(l)
	for err != io.EOF {
		if err == nil {
			fmt.Println("eval:", expr.eval())
		} else {
			fmt.Println("err:", err)
		}
		expr, err = parseLine(l)
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
