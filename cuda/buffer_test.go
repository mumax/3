package cuda

import "testing"

// In case of memory leak, this will crash
func TestBuffer(t *testing.T) {
	m1 := [3]int{2, 1024, 2048}
	m2 := [3]int{4, 1024, 2048}
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
	m := [3]int{2, 1024, 2048}
	a := Buffer(3, m)
	Recycle(a)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		a := Buffer(3, m)
		Recycle(a)
	}
}
