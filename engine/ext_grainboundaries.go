package engine

import (
	"math"
)

var (
	ext_grainboundary_edgeX bool = true
	ext_grainboundary_edgeY bool = true
	ext_grainboundary_edgeZ bool = false
)

func init() {
	DeclFunc("ext_grainboundaries", ext_grainboundaries, "(startregion, numgrains, offset, boundarythickness, zeroflag). Given existing regions, reassigns grain boundaries of boundarythickness to new region values, starting at offset. Zeroflag: 1 = region0 is normal, 0 = region0 acts as edge but no boundary itself, -1 = ignore region0 entirely. grainboundary_edgeX/Y/Z control whether simulatiion box edges are treated as grainboundaries.")
	DeclVar("ext_grainboundary_edgeX", &ext_grainboundary_edgeX, "Treat X edges of simulation box as boundaries. Ignored if PBC in X direction enabled (default= true)")
	DeclVar("ext_grainboundary_edgeY", &ext_grainboundary_edgeY, "Treat Y edges of simulation box as boundaries. Ignored if PBC in Y direction enabled (default= true)")
	DeclVar("ext_grainboundary_edgeZ", &ext_grainboundary_edgeZ, "Treat Z edges of simulation box as boundaries. Ignored if PBC in Z direction enabled (default= false)")
}

func ext_grainboundaries(startregion, numgrains, offset int, boundarythickness float64, zeroflag int) {
	r := &regions
	mesh := r.Mesh()
	size := mesh.Size()
	Nx, Ny, Nz := size[X], size[Y], size[Z]
	cellsize := mesh.CellSize()
	cx, cy, cz := cellsize[X], cellsize[Y], cellsize[Z]

	host := r.HostList()
	orig := make([]byte, len(host))
	copy(orig, host)
	origArr := reshapeBytes(orig, size)

	rx, ry, rz := boundarythickness/cx, boundarythickness/cy, boundarythickness/cz
	Rx, Ry, Rz := int(rx), int(ry), int(rz)                      // Round down to nearest integer for optimal for loops
	rxSqInv, rySqInv, rzSqInv := 1/(rx*rx), 1/(ry*ry), 1/(rz*rz) // Axes of the circle in cell lengths (squared inverse)

	dx := make([]int, 0, (2*Rx+1)*(2*Ry+1)*(2*Rz+1))
	dy := make([]int, 0, (2*Rx+1)*(2*Ry+1)*(2*Rz+1))
	dz := make([]int, 0, (2*Rx+1)*(2*Ry+1)*(2*Rz+1))
	for k := -Rz; k <= Rz; k++ {
		for j := -Ry; j <= Ry; j++ {
			for i := -Rx; i <= Rx; i++ {
				if float64(i*i)*rxSqInv+float64(j*j)*rySqInv+float64(k*k)*rzSqInv <= 1. {
					dx = append(dx, i)
					dy = append(dy, j)
					dz = append(dz, k)
				}
			}
		}
	}

	for iz := 0; iz < Nz; iz++ {
		for iy := 0; iy < Ny; iy++ {
			for ix := 0; ix < Nx; ix++ {
				reg := int(origArr[iz][iy][ix])

				if (zeroflag == -1 && reg == 0) || (zeroflag == 0 && reg == 0) {
					continue
				}

				if reg < startregion || reg > startregion+numgrains {
					continue
				}

				isBoundary := false
				for i := 0; i < len(dx); i++ {
					nx := wrapPBC(ix+dx[i], X)
					ny := wrapPBC(iy+dy[i], Y)
					nz := wrapPBC(iz+dz[i], Z)

					outX := nx < 0 || nx >= Nx
					outY := ny < 0 || ny >= Ny
					outZ := nz < 0 || nz >= Nz

					if outX || outY || outZ {
						if (outX && ext_grainboundary_edgeX) || (outY && ext_grainboundary_edgeY) || (outZ && ext_grainboundary_edgeZ) {
							isBoundary = true
							break
						}
						continue
					}

					neighbor := int(origArr[nz][ny][nx])
					if zeroflag == -1 && neighbor == 0 {
						continue
					}
					if neighbor != reg {
						isBoundary = true
						break
					}
				}

				if isBoundary {
					host[iz*Ny*Nx+iy*Nx+ix] = byte(reg + offset)
				}
			}
		}
	}

	// Upload updated host array to GPU
	r.gpuCache.Upload(host)
	arr := reshapeBytes(host, size)
	f := func(x, y, z float64) int {
		ix := floatToIndex(x, Nx)
		iy := floatToIndex(y, Ny)
		iz := floatToIndex(z, Nz)
		return int(arr[iz][iy][ix])
	}
	r.hist = append(r.hist, f)
}

func floatToIndex(x float64, N int) int {
	ix := int(math.Round(x))
	if ix < 0 {
		ix = 0
	}
	if ix >= N {
		ix = N - 1
	}
	return ix
}
