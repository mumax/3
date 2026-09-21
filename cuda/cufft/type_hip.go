//go:build hip

package cufft

//#include <hipfft/hipfft.h>
import "C"

import (
	"fmt"
)

// FFT type
type Type int

const (
	R2C Type = C.HIPFFT_R2C // Real to Complex (interleaved)
	C2R Type = C.HIPFFT_C2R // Complex (interleaved) to Real
	C2C Type = C.HIPFFT_C2C // Complex to Complex, interleaved
	D2Z Type = C.HIPFFT_D2Z // Double to Double-Complex
	Z2D Type = C.HIPFFT_Z2D // Double-Complex to Double
	Z2Z Type = C.HIPFFT_Z2Z // Double-Complex to Double-Complex
)

const (
	FORWARD = -1 // Forward FFT
	INVERSE = 1  // Inverse FFT
)

func (t Type) String() string {
	if str, ok := typeString[t]; ok {
		return str
	}
	return fmt.Sprint("FFT Type with unknown number:", int(t))
}

var typeString = map[Type]string{
	R2C: "HIPFFT_R2C",
	C2R: "HIPFFT_C2R",
	C2C: "HIPFFT_C2C",
	D2Z: "HIPFFT_D2Z",
	Z2D: "HIPFFT_Z2D",
	Z2Z: "HIPFFT_Z2Z"}
