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
	p := NewParser(src)
	p.AddFloat("a", &a)
	Expr, err := p.parseLine()
	for err != io.EOF {
		if err == nil {
			fmt.Println("eval", Expr, ":", Expr.Eval())
		} else {
			fmt.Println("err:", err)
		}
		Expr, err = p.parseLine()
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
