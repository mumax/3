package engine

import (
	"math"

	"github.com/mumax/3/util"
)

func init() {
	DeclFunc("ext_grainboundaries", ext_grainboundaries, "(startregion, numgrains, offset, boundarythickness, zeroflag). Given existing regions, reassigns grain boundaries of boundarythickness to new region values, starting at offset. Zeroflag: 1 = region0 is normal, 0 = region0 acts as edge but no boundary itself, -1 = ignore region0 entirely.")
}

func ext_grainboundaries(startregion, numgrains, offset, boundarythickness, zeroflag int) {
	r := &regions

	size := r.Mesh().Size()
	Nx, Ny, Nz := size[X], size[Y], size[Z]

	host := r.HostList()

	orig := make([]byte, len(host))
	copy(orig, host)
	origArr := reshapeBytes(orig, size)

	if boundarythickness < 1 {
		util.Log("boundarythickness must be >= 1")
		return
	}

	R := boundarythickness

	dx := make([]int, 0, (2*R+1)*(2*R+1)*(2*R+1))
	dy := make([]int, 0, (2*R+1)*(2*R+1)*(2*R+1))
	dz := make([]int, 0, (2*R+1)*(2*R+1)*(2*R+1))

	for k := -R; k <= R; k++ {
		for j := -R; j <= R; j++ {
			for i := -R; i <= R; i++ {
				if i*i+j*j+k*k <= R*R {
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
					nx := ix + dx[i]
					ny := iy + dy[i]
					nz := iz + dz[i]

					if nx < 0 || nx >= Nx || ny < 0 || ny >= Ny || nz < 0 || nz >= Nz {
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
