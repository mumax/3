package cuda

import (
	"code.google.com/p/mx3/data"
	"testing"
)

func init() {
	Init(0, "auto")
}

func TestBuffer(t *testing.T) {
	C := 1e-9
	m1 := data.NewMesh(2, 1024, 2048, C, C, C)
	m2 := data.NewMesh(4, 1024, 2048, C, C, C)
	a := GetBuffer(3, m1)
	b := GetBuffer(3, m2)
	c := GetBuffer(1, m1)
	d := GetBuffer(2, m2)

	RecycleBuffer(a)
	RecycleBuffer(b)
	RecycleBuffer(c)
	RecycleBuffer(d)

	for i := 0; i < 10000; i++ {
		b := GetBuffer(3, m2)
		RecycleBuffer(b)
	}
}

func BenchmarkBuffer(b *testing.B) {
	b.StopTimer()
	C := 1e-9
	m := data.NewMesh(2, 1024, 2048, C, C, C)
	a := GetBuffer(3, m)
	RecycleBuffer(a)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		a := GetBuffer(3, m)
		RecycleBuffer(a)
	}
}
