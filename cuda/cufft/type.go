package cufft

//#include <cufft.h>
import "C"

import (
	"fmt"
)

// FFT type
type Type int

const (
	R2C Type = C.CUFFT_R2C // Real to Complex (interleaved)
	C2R Type = C.CUFFT_C2R // Complex (interleaved) to Real
	C2C Type = C.CUFFT_C2C // Complex to Complex, interleaved
	D2Z Type = C.CUFFT_D2Z // Double to Double-Complex
	Z2D Type = C.CUFFT_Z2D // Double-Complex to Double
	Z2Z Type = C.CUFFT_Z2Z // Double-Complex to Double-Complex
)

const (
	FORWARD = -1 // Forward FFT
	INVERSE = 1  // Inverse FFT
)

func (t Type) String() string {
	if str, ok := typeString[t]; ok {
		return str
	}
	return fmt.Sprint("CUFFT Type with unknown number:", int(t))
}

var typeString map[Type]string = map[Type]string{
	R2C: "CUFFT_R2C",
	C2R: "CUFFT_C2R",
	C2C: "CUFFT_C2C",
	D2Z: "CUFFT_D2Z",
	Z2D: "CUFFT_Z2D",
	Z2Z: "CUFFT_Z2Z"}
