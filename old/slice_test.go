package nimble

import (
	_ "code.google.com/p/mx3/gpu"
	"testing"
)

func TestSlice(t *testing.T) {
	a := make([]float32, 100)
	s := makeSlice(1, 100, GPUMemory)
	if len(a) != s.Len() {
		t.Fail()
	}

	a = a[10:42]
	s = s.Slice(10, 42)
	if len(a) != s.Len() {
		t.Fail()
	}
}
