package cuda

import (
	"github.com/mumax/3/data"
	"testing"
)

func init() {
	Init(0, "auto")
}

func TestBuffer(t *testing.T) {
	C := 1e-9
	m1 := data.NewMesh(2, 1024, 2048, C, C, C)
	m2 := data.NewMesh(4, 1024, 2048, C, C, C)
	a := Buffer(3, m1)
	b := Buffer(3, m2)
	c := Buffer(1, m1)
	d := Buffer(2, m2)

	Recycle(a)
	Recycle(b)
	Recycle(c)
	Recycle(d)

	for i := 0; i < 10000; i++ {
		b := Buffer(3, m2)
		Recycle(b)
	}
}

func BenchmarkBuffer(b *testing.B) {
	b.StopTimer()
	C := 1e-9
	m := data.NewMesh(2, 1024, 2048, C, C, C)
	a := Buffer(3, m)
	Recycle(a)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		a := Buffer(3, m)
		Recycle(a)
	}
}
