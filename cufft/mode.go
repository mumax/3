package cufft

//#include <cufft.h>
import "C"

import (
	"fmt"
)

// CUFFT compatibility mode
type CompatibilityMode int

const (
	COMPATIBILITY_NATIVE          CompatibilityMode = C.CUFFT_COMPATIBILITY_NATIVE
	COMPATIBILITY_FFTW_PADDING    CompatibilityMode = C.CUFFT_COMPATIBILITY_FFTW_PADDING
	COMPATIBILITY_FFTW_ASYMMETRIC CompatibilityMode = C.CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC
	COMPATIBILITY_FFTW_ALL        CompatibilityMode = C.CUFFT_COMPATIBILITY_FFTW_ALL
)

func (t CompatibilityMode) String() string {
	if str, ok := compatibilityModeString[t]; ok {
		return str
	}
	return fmt.Sprint("CUFFT Compatibility mode with unknown number:", int(t))
}

var compatibilityModeString map[CompatibilityMode]string = map[CompatibilityMode]string{
	COMPATIBILITY_NATIVE:          "CUFFT_COMPATIBILITY_NATIVE",
	COMPATIBILITY_FFTW_PADDING:    "CUFFT_COMPATIBILITY_FFTW_PADDING",
	COMPATIBILITY_FFTW_ASYMMETRIC: "CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC",
	COMPATIBILITY_FFTW_ALL:        "CUFFT_COMPATIBILITY_FFTW_ALL"}
