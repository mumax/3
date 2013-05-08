package script

import (
	"bytes"
	"fmt"
	"io"
	"testing"
)

var (
	a float64
)

func TestParser(t *testing.T) {
	src := bytes.NewBuffer([]byte(testText))
	p := newParser(src)
	p.addFloat("a", &a)
	expr, err := p.ParseLine()
	for err != io.EOF {
		if err == nil {
			fmt.Println("eval", expr, ":", expr.eval())
		} else {
			fmt.Println("err:", err)
		}
		expr, err = p.ParseLine()
	}
}

const testText = `
	a
	a=1
	a
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
