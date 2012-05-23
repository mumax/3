package nc

import (
	"math"
	"testing"
)

func TestSqrtf(t *testing.T) {
	f := Sqrtf(2)
	if f != 1.41421356237 {
		t.Error("Sqrtf(2):", f)
	}
	f = Sqrtf(-1)
	if !math.IsNaN(float64(f)) {
		t.Error("Sqrtf(-1):", f)
	}
}
