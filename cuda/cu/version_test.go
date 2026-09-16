//go:build !darwin || !arm64
// +build !darwin !arm64

package cu

import (
	"fmt"
	"testing"
)

func TestVersion(t *testing.T) {
	fmt.Println("CUDA driver version: ", Version())
}
