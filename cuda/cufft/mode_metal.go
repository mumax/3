//go:build darwin && arm64
// +build darwin,arm64

package cufft

import "fmt"

type CompatibilityMode int

const COMPATIBILITY_FFTW_PADDING CompatibilityMode = 1

func (t CompatibilityMode) String() string {
	if str, ok := compatibilityModeString[t]; ok {
		return str
	}
	return fmt.Sprint("CUFFT Compatibility mode with unknown number:", int(t))
}

var compatibilityModeString = map[CompatibilityMode]string{
	COMPATIBILITY_FFTW_PADDING: "CUFFT_COMPATIBILITY_FFTW_PADDING",
}
