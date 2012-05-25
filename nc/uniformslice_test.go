package nc

import (
	"math/rand"
	"testing"
)

func TestUniformSlice(test *testing.T) {
	b := MakeUniformSlice()

	for i := 0; i < 1000; i++ {
		i1 := rand.Int() % 10
		i2 := i1 + rand.Int()%10
		value := float32(rand.Int() % 3)
		b.SetValue(value)
		Range := b.Range(i1, i2)
		if len(Range) != i2-i1 {
			test.Error(len(Range), "!=", i2-i1)
		}
		for i := range Range {
			if Range[i] != value {
				test.Error(Range, "!=", value)
			}
		}
	}
}
