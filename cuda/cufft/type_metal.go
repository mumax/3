//go:build darwin && arm64
// +build darwin,arm64

package cufft

import "fmt"

// Type retains cuFFT's numeric ABI values for source compatibility.
type Type int

const (
	R2C Type = 0x2a
	C2R Type = 0x2c
	C2C Type = 0x29
	D2Z Type = 0x6a
	Z2D Type = 0x6c
	Z2Z Type = 0x69
)

const (
	FORWARD = -1
	INVERSE = 1
)

func (t Type) String() string {
	if str, ok := typeString[t]; ok {
		return str
	}
	return fmt.Sprint("CUFFT Type with unknown number:", int(t))
}

var typeString = map[Type]string{
	R2C: "CUFFT_R2C",
	C2R: "CUFFT_C2R",
	C2C: "CUFFT_C2C",
	D2Z: "CUFFT_D2Z",
	Z2D: "CUFFT_Z2D",
	Z2Z: "CUFFT_Z2Z",
}
