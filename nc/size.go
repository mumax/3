package nc

// Globals

import ()

var (
	size      [3]int // 3D geom size
	n         int    // product of size
	nWarp     int    // number of slices
	warpSize  [3]int // slice size
	warpLen   int    // product of slice size
	warpPitch [3]int
	cellSize  [3]float64
	periodic  [3]int
)

var MAX_WARPLEN = 8192 // elements. 

func InitCellSize(size0, size1, size2 float64) {
	cellSize = [3]float64{size0, size1, size2}
}

func InitSize(N0, N1, N2 int) {
	Assert(N2 > 0)
	Assert(N1 > 0)
	Assert(N0 > 0)

	size = [3]int{N0, N1, N2}
	n = N0 * N1 * N2

	Log("Size:", size)
	Debug("N:", n)
	Debug("Max warpLen:", MAX_WARPLEN)

	minNw := 1       // need min. 1 warp
	maxNw := N0 * N1 // max. Nwarp: do not slice up along K, keep full rows.

	nWarp = maxNw
	for w := maxNw; w >= minNw; w-- {
		if N0%w != 0 && N1%w != 0 { // need to slice along either I or J
			continue
		}
		if n/w > MAX_WARPLEN { // warpLen must not be larger than specified.
			break
		}
		nWarp = w
	}
	warpLen = n / nWarp

	if nWarp <= N0 { // slice along I
		warpSize = [3]int{N0 / nWarp, N1, N2}
	} else { // slice along I and J 
		warpSize = [3]int{1, (N0 * N1) / nWarp, N2}
	}

	Debug("NumWarp:", nWarp)
	Debug("WarpLen:", warpLen)
	Debug("WarpSize:", warpSize)

	Assert(WarpSize()[0]*WarpSize()[1]*WarpSize()[2] == WarpLen())
	Assert(Size()[0]*Size()[1]*Size()[2] == N())
	Assert(WarpLen()*NumWarp() == N())

	InitGarbageman()
}

func DefaultBufSize() int {
	return nWarp
}

func Size() [3]int { return size }

func CellSize() [3]float64 { return cellSize }

func N() int { return n }

func WarpLen() int { return warpLen }

func NumWarp() int { return nWarp }

func WarpSize() [3]int { return warpSize }

func Periodic() [3]int { return periodic }

// Position of 3D slice number s in its full 3D block.
func SliceOffset(s int) [3]int {
	N0 := size[0]
	N1 := size[1]
	if nWarp <= N0 { // slice along I
		i := s * (N0 / nWarp)
		return [3]int{i, 0, 0}
	} //else
	j := s * (N1 / nWarp)
	i := j / (N0 * N1)
	j %= N1
	return [3]int{i, j, 0}
}

// Zero-padded size.
func PadSize(size [3]int) [3]int {
	padded := [3]int{
		size[0] * 2,
		size[1] * 2,
		size[2] * 2}
	if padded[0] == 2 {
		padded[0] = 1 // no need to pad 1 layer thickness
	}
	return padded
}
