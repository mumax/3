package mx

import (
	"fmt"
	"testing"
)

func TestSum(t *testing.T) {
	a := NewUniformScalar(42)
	b := NewUniform([]float32{1})
	sum := NewSum(a, b)
	if sum.IGet(0, 0) != 43 {
		t.Fail()
	}
}

func TestBounds(t *testing.T) {
	defer func() {
		err := recover()
		if err == nil {
			t.Fail()
		}
	}()
	a := NewUniformScalar(42)
	b := NewUniform([]float32{1})
	sum := NewSum(a, b)
	fmt.Println(QString(sum), sum.IGet(1, 0)) // must go out of bounds 
}

func BenchmarkSum(b *testing.B) {
	b.StopTimer()
	A := NewUniformScalar(42)
	B := NewUniform([]float32{1})
	sum := NewSum(A, B)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		sum.IGet(0, 0)
	}
}

func BenchmarkGoSum(b *testing.B) {
	var A, B, C float32
	for i := 0; i < b.N; i++ {
		C = A + B
	}
	println(C)
}
