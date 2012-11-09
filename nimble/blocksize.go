package nimble

// BlockSize finds a suitable way to split an array
// of given size into equal blocks.
// It limits block sizes to Flag_maxblocklen.
func BlockSize(size [3]int) [3]int {
	N0, N1, N2 := size[0], size[1], size[2]
	n := prod(size)

	minNw := *Flag_minblocks // minimum number of blocks
	maxNw := N0 * N1         // max. Nwarp: do not slice up along K, keep full rows.

	nWarp := maxNw
	for w := maxNw; w >= minNw; w-- {
		if N0%w != 0 && N1%w != 0 { // need to slice along either I or J
			continue
		}
		if n/w > *Flag_maxblocklen { // warpLen must not be larger than specified.
			break
		}
		nWarp = w
	}

	if nWarp <= N0 { // slice along I
		return [3]int{N0 / nWarp, N1, N2}
	} // else { // slice along I and J 
	return [3]int{1, (N0 * N1) / nWarp, N2}
}

func prod(size [3]int) int {
	return size[0] * size[1] * size[2]
}
