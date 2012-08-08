package nc

// Utilities for loading PTX assembly code.

// Directory where to look for PTX files.
func PtxDir() string {
	return "/home/arne/go/src/nimble-cube/ptx"
	//return ExecutableDir() + "../src/nimble-cube/ptx/" // KO with go run
}

// Integer division rounded up.
func DivUp(x, y int) int {
	return ((x - 1) / y) + 1
}
