package mx

import (
	"fmt"
	"testing"
)

func TestSum(t *testing.T) {
	a := NewUniformScalar(42)
	b := NewUniform([]float32{1})
	sum := NewSum(a, b)
	fmt.Println(QString(sum), sum.IGet(0, 0))
}
