package script

import (
	"bytes"
	"testing"
)

var (
	a float64
)

func TestParser(t *testing.T) {
	src1 := bytes.NewBuffer([]byte(testText))
	p := NewParser()
	p.AddFloat("a", &a)
	p.Exec(src1)
	p.ExecString("a=2")
}

const testText = `
	a
	a=1
	a
	a() 
	a(1)
`
