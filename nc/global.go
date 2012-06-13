package nc

// Globals

import ()

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

	Log("Size:", size)
	Debug("N:", n)

	// Find some nice warp size
	warp = MAX_WARP
	Assert(warp >= 1)
	for n%warp != 0 {
		warp--
	}
	Debug("WarpLen:", warp)
	nWarp = n / warp
	Debug("NumWarp:", nWarp)
}

func DefaultBufSize() int {
	return nWarp
}

func Size() [3]int { return size }

func N() int { return n }

func WarpLen() int { return warp }

func NumWarp() int { return nWarp }
