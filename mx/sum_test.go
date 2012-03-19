package mx

import (
	"fmt"
	"testing"
)

func TestSum(t *testing.T) { //←[ TestSum t does not escape]
	a := NewUniformScalar(42)
	b := NewUniform([]float32{1}) //←[ TestSum []float32 literal does not escape]
	sum := NewSum(a, b)
	fmt.Println(QString(sum), sum.IGet(0, 0)) //←[ TestSum ... argument does not escape]
}

func BenchmarkSum(b *testing.B){//←[ BenchmarkSum b does not escape]
	b.StopTimer()
	A := NewUniformScalar(42)
	B := NewUniform([]float32{1}) //←[ BenchmarkSum []float32 literal does not escape]
	sum := NewSum(A, B)
	b.StartTimer()
	for i:=0; i<b.N; i++{
		sum.IGet(0, 0)
	}
}

func BenchmarkGoSum(b*testing.B){//←[ BenchmarkGoSum b does not escape]
	var A,B,C float32
	for i:=0; i<b.N; i++{
		C = A + B
	}
	println(C)
}
