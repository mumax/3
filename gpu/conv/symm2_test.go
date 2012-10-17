package conv

import (
	"testing"
)

var (
	N0s = []int{1}
	N1s = []int{2, 3, 4, 8, 16, 32, 48, 63, 64, 65}
	N2s = []int{2, 3, 4, 8, 16, 32, 48, 64, 128, 255, 256, 257, 1024}
)

func TestSymmetric2(t *testing.T) {
	for _, N0 := range N0s {
		for _, N1 := range N1s {
			for _, N2 := range N2s {
				TestSymm2(N0, N1, N2)
			}
		}
	}
}
