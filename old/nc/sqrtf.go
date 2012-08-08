package nc

import (
	"math"
)

// Portable sqrtf.
func Sqrtf(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}
