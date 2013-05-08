package script

import (
	"bytes"
	"testing"
)

var (
	a float64
)

func TestParser(t *testing.T) {
	src := bytes.NewBuffer([]byte(testText))
	p := NewParser(src)
	p.AddFloat("a", &a)
	p.Exec()
}

const testText = `
	a
	a=1
	a
	a() 
	a(1)
`
