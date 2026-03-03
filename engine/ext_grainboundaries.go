package engine

func init() {
	DeclFunc("ext_grainboundaries", ext_grainboundaries, "Mark grain boundary cells as a separate region")
}

// Precompute offsets within a radius (for expanding boundary zone)
func precomputeOffsetsArray(borderSize int) [][3]int {
	var offsets [][3]int
	radiusSq := borderSize * borderSize
	for kk := -borderSize + 1; kk < borderSize; kk++ {
		for jj := -borderSize + 1; jj < borderSize; jj++ {
			for ii := -borderSize + 1; ii < borderSize; ii++ {
				if ii*ii+jj*jj+kk*kk <= radiusSq {
					offsets = append(offsets, [3]int{kk, jj, ii})
				}
			}
		}
	}
	return offsets
}

func ext_grainboundaries(numgrains, boundarythickness, zeroflag int) {
	r := &regions

	host := r.HostList()
	size := r.Mesh().Size()
	arr := reshapeBytes(host, size)

	Nx, Ny, Nz := size[X], size[Y], size[Z]

	// Snapshot original regions
	orig := make([]byte, len(host))
	copy(orig, host)
	origArr := reshapeBytes(orig, size)

	// Precompute Chebyshev offsets for boundary growth
	var offsets [][3]int
	for dz := -boundarythickness + 1; dz <= boundarythickness-1; dz++ {
		for dy := -boundarythickness + 1; dy <= boundarythickness-1; dy++ {
			for dx := -boundarythickness + 1; dx <= boundarythickness-1; dx++ {
				offsets = append(offsets, [3]int{dz, dy, dx})
			}
		}
	}

	for iz := 0; iz < Nz; iz++ {
		for iy := 0; iy < Ny; iy++ {
			for ix := 0; ix < Nx; ix++ {

				region := origArr[iz][iy][ix]

				// Skip region 0 if zeroflag == 0 or -1
				if region == 0 && (zeroflag == 0 || zeroflag == -1) {
					continue
				}

				isBoundary := false

				checkNeighbor := func(nz, ny, nx int) bool {
					neighbor := origArr[nz][ny][nx]

					// If zeroflag == 0, ignore zero neighbors
					if zeroflag == 0 && neighbor == 0 {
						return false
					}

					// Otherwise, any neighbor not equal to region counts
					return neighbor != region
				}

				// 6-neighbor boundary check
				if iz > 0 && checkNeighbor(iz-1, iy, ix) {
					isBoundary = true
				}
				if !isBoundary && iz < Nz-1 && checkNeighbor(iz+1, iy, ix) {
					isBoundary = true
				}
				if !isBoundary && iy > 0 && checkNeighbor(iz, iy-1, ix) {
					isBoundary = true
				}
				if !isBoundary && iy < Ny-1 && checkNeighbor(iz, iy+1, ix) {
					isBoundary = true
				}
				if !isBoundary && ix > 0 && checkNeighbor(iz, iy, ix-1) {
					isBoundary = true
				}
				if !isBoundary && ix < Nx-1 && checkNeighbor(iz, iy, ix+1) {
					isBoundary = true
				}

				if isBoundary {
					// Grow boundary cube only inside the same original region
					for _, off := range offsets {
						nz := iz + off[0]
						ny := iy + off[1]
						nx := ix + off[2]

						if nz < 0 || nz >= Nz || ny < 0 || ny >= Ny || nx < 0 || nx >= Nx {
							continue
						}

						// Only update cells belonging to the original region
						if origArr[nz][ny][nx] == region {
							host[nz*Nx*Ny+ny*Nx+nx] = byte(int(region) + numgrains)
						}
					}
				}
			}
		}
	}

	r.gpuCache.Upload(host)

	// Save history
	f := func(x, y, z float64) int {
		ix := floatToIndex(x, Nx)
		iy := floatToIndex(y, Ny)
		iz := floatToIndex(z, Nz)
		return int(arr[iz][iy][ix])
	}
	r.hist = append(r.hist, f)
}

func floatToIndex(x float64, N int) int {
	ix := int(x + 0.5) // round to nearest voxel
	if ix < 0 {
		ix = 0
	}
	if ix >= N {
		ix = N - 1
	}
	return ix
}
