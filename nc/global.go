package nc

// Globals

import (
	"log"
)

var (
	size  [3]int // 3D geom size
	n     int    // product of size
	warp  int    // slice size
	nWarp int    // slice size
)

var MAX_WARP = 8192

func InitSize(N0, N1, N2 int) {
	size = [3]int{N0, N1, N2}
	n = N0 * N1 * N2

	log.Println("Size:", size)
	log.Println("N:", n)

	// Find some nice warp size
	warp = MAX_WARP
	Assert(warp >= 1)
	for n%warp != 0 {
		warp--
	}
	log.Println("WarpLen:", warp)
	nWarp = n / warp
	log.Println("NumWarp:", nWarp)

	initGarbageman()

}

func DefaultBufSize() int {
	return nWarp
}

func Size() [3]int { return size }

func N() int { return n }

func WarpLen() int { return warp }

func NumWarp() int { return nWarp }
