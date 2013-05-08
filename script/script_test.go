package script

import (
	"bytes"
	"fmt"
	"testing"
)

var (
	a float64
)

func TestParser(t *testing.T) {
	src1 := bytes.NewBuffer([]byte(testText))
	p := NewParser()
	p.AddFloat("a", &a)
	p.AddFunc("print", myprint)
	p.Exec(src1)
	p.ExecString("a=2; print(a, 2)")
}

func myprint(msg ...interface{}) {
	fmt.Println(msg...)
}

const testText = `
	a
	a=12e-13
	a
	a(a()) 
	a(1)
`

func BenchmarkParser(b *testing.B) {
	b.StopTimer()
	b.SetBytes(int64(len(testText)))
	src := bytes.NewBuffer([]byte(testText))
	p := NewParser()
	p.AddFloat("a", &a)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		p.parse(src)
	}
}
