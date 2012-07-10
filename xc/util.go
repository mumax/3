package xc

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Integer division rounded up.
func DivUp(x, y int) int {
	return ((x - 1) / y) + 1
}
