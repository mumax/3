package mag

import (
	"fmt"
	"math"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// FasterPBCKernel selects the demag kernel method. Set from the input script, case-insensitive:
//
//	FasterPBCKernel = "off"   // or "old": original CalcDemagKernel (default)
//	FasterPBCKernel = "CPU"   // CalcDemagKernelFast: all CPU cores, float64 image sums
//
// "CPU" falls back to the original method (logged) if there is not enough memory for its image cache.
var FasterPBCKernel = "off"

// kernelMethod normalizes a FasterPBCKernel value to "old" or "cpu".
func kernelMethod(v string) (string, error) {
	switch m := strings.ToLower(strings.TrimSpace(v)); m {
	case "off", "old", "":
		return "old", nil
	case "cpu":
		return m, nil
	}
	return "", fmt.Errorf(`FasterPBCKernel = %q: use "off", "old" or "CPU"`, v)
}

// calcKernel dispatches on FasterPBCKernel.
//
// Migration plan: (1) now: opt-in, default "off"; (2) later: make "CPU" the default above;
// (3) finally: delete CalcDemagKernel (demagkernel.go) and the "old" branch and fallback below; for the
// low-memory case, let CalcDemagKernelFast run with mirror=false instead of returning ok=false.
// Nothing in the new code depends on CalcDemagKernel; the tests use it only as a reference.
func calcKernel(inputSize, pbc [3]int, cellsize [3]float64, accuracy float64) [3][3]*data.Slice {
	method, err := kernelMethod(FasterPBCKernel)
	if err != nil {
		util.Fatal(err)
	}
	if method == "cpu" {
		if k, ok := CalcDemagKernelFast(inputSize, pbc, cellsize, accuracy, availableMemory()/2); ok {
			return k
		}
		util.Log("//FasterPBCKernel: not enough memory, falling back to standard kernel calculation")
	}
	return CalcDemagKernel(inputSize, pbc, cellsize, accuracy)
}

// CalcDemagKernelFast computes the same kernel as CalcDemagKernel, but
//   - parallelizes over all CPU cores instead of 3 goroutines (one per source component),
//   - for size[Y]==1 (thin film, PBC in Y) computes each y-image once for y>=0 and
//     reuses it for -y via mirror symmetry (2x fewer evaluations).
//
// Work is split per destination cell. Each cell sums its images in the order of CalcDemagKernel
// (z, then y, then x images, ascending), but in float64, rounded to float32 once at the end.
// CalcDemagKernel sums in float32, which for many PBC images loses ~1e-6 of max|N| and silently
// drops far images once they fall below float32 resolution of the running sum.
// With the mirror trick, the mirrored addends can differ from directly computed ones by float64
// rounding only.
//
// memBudget (bytes) bounds the extra per-worker image cache; ok=false if not even one worker fits.
func CalcDemagKernelFast(inputSize, pbc [3]int, cellsize [3]float64, accuracy float64, memBudget int64) (kernel [3][3]*data.Slice, ok bool) {
	size := padSize(inputSize, pbc)
	util.Assert(size[Z] > 0 && size[Y] > 0 && size[X] > 0)
	util.Assert(cellsize[X] > 0 && cellsize[Y] > 0 && cellsize[Z] > 0)
	util.Assert(pbc[X] >= 0 && pbc[Y] >= 0 && pbc[Z] >= 0)
	util.Assert(accuracy > 0)

	r1, r2 := kernelRanges(size, pbc)
	L := math.Min(cellsize[X], math.Min(cellsize[Y], cellsize[Z]))

	// all loop coordinates in [r1[c], r2[c]] that wrap onto target index t
	images := func(c, t int) []int {
		var im []int
		for v := r1[c] + ((t-r1[c])%size[c]+size[c])%size[c]; v <= r2[c]; v += size[c] {
			im = append(im, v)
		}
		return im
	}

	// Mirror y -> -y keeps the target cell only if every y image maps to the same target.
	mirror := size[Y] == 1 && r1[Y] == -r2[Y]
	nxMax := (r2[X]-r1[X])/size[X] + 2
	cacheLen := 0
	if mirror {
		cacheLen = (r2[Y] + 1) * nxMax * 3
	}
	perWorker := int64(cacheLen) * 8
	workers := runtime.GOMAXPROCS(0)
	if perWorker > 0 {
		if memBudget < perWorker {
			return kernel, false
		}
		if int64(workers) > memBudget/perWorker {
			workers = int(memBudget / perWorker)
		}
	}

	var array [3][3][][][]float32
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			kernel[i][j] = data.NewSlice(1, size)
			array[i][j] = kernel[i][j].Scalars()
		}
	}

	// one job = one (source component s, destination cell) in the non-skipped half
	hx, hy, hz := size[X]/2+1, size[Y]/2+1, size[Z]/2+1
	nJobs := int64(3 * hz * hy * hx)
	util.Log(fmt.Sprintf("//FasterPBCKernel: %d workers, y-mirror: %v", workers, mirror))
	util.Progress(0, int(nJobs), "Calculating demag kernel (fast)")

	var next, done int64 = -1, 0
	var wg sync.WaitGroup
	for wk := 0; wk < workers; wk++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			cache := make([]float64, cacheLen)
			for {
				j := atomic.AddInt64(&next, 1)
				if j >= nJobs {
					return
				}
				xw := int(j % int64(hx))
				yw := int(j / int64(hx) % int64(hy))
				zw := int(j / int64(hx*hy) % int64(hz))
				s := int(j / int64(hx*hy*hz))
				xs, ys, zs := images(X, xw), images(Y, yw), images(Z, zw)

				var acc [3]float64 // float64 sum over images, rounded to float32 once at the end
				for _, z := range zs {
					if mirror {
						for _, y := range ys {
							if y < 0 {
								continue
							}
							for ix, x := range xs {
								B := kernelB(s, x, y, z, cellsize, accuracy, L)
								copy(cache[(y*nxMax+ix)*3:], B[:])
							}
						}
					}
					for _, y := range ys {
						for ix, x := range xs {
							var B [3]float64
							switch {
							case !mirror:
								B = kernelB(s, x, y, z, cellsize, accuracy, L)
							case y >= 0:
								copy(B[:], cache[(y*nxMax+ix)*3:])
							default: // y<0: mirror of -y; components with exactly one Y index flip sign
								copy(B[:], cache[(-y*nxMax+ix)*3:])
								for d := 0; d < 3; d++ {
									if (s == Y) != (d == Y) {
										B[d] = -B[d]
									}
								}
							}
							for d := s; d < 3; d++ {
								acc[d] += B[d]
							}
						}
					}
				}
				for d := s; d < 3; d++ {
					array[s][d][zw][yw][xw] = float32(acc[d])
				}
				if n := atomic.AddInt64(&done, 1); n%1024 == 0 {
					util.Progress(int(n), int(nJobs), "Calculating demag kernel (fast)")
				}
			}
		}()
	}
	wg.Wait()

	return symmetrizeKernel(kernel, array, size), true
}

// kernelB: field B at destination cell offset (x,y,z) (in cells) from source charges on the
// faces of a unit-magnetized cell perpendicular to s. Same arithmetic as the body of CalcDemagKernel.
func kernelB(s, x, y, z int, cellsize [3]float64, accuracy, L float64) (B [3]float64) {
	u, v, w := s, (s+1)%3, (s+2)%3
	var R, R2, pole [3]float64
	R[X], R[Y], R[Z] = float64(x)*cellsize[X], float64(y)*cellsize[Y], float64(z)*cellsize[Z]

	dx, dy, dz := delta(x)*cellsize[X], delta(y)*cellsize[Y], delta(z)*cellsize[Z]
	d := math.Sqrt(dx*dx + dy*dy + dz*dz)
	if d == 0 {
		d = L
	}
	maxSize := d / accuracy

	nv := int(math.Max(cellsize[v]/maxSize, 1) + 0.5)
	nw := int(math.Max(cellsize[w]/maxSize, 1) + 0.5)
	nx := int(math.Max(cellsize[X]/maxSize, 1) + 0.5)
	ny := int(math.Max(cellsize[Y]/maxSize, 1) + 0.5)
	nz := int(math.Max(cellsize[Z]/maxSize, 1) + 0.5)
	nv *= 2
	nw *= 2

	scale := 1 / float64(nv*nw*nx*ny*nz)
	surface := cellsize[v] * cellsize[w]
	charge := surface * scale
	pu1 := cellsize[u] / 2.
	pu2 := -pu1

	for i := 0; i < nv; i++ {
		pole[v] = -(cellsize[v] / 2.) + cellsize[v]/float64(2*nv) + float64(i)*(cellsize[v]/float64(nv))
		for j := 0; j < nw; j++ {
			pole[w] = -(cellsize[w] / 2.) + cellsize[w]/float64(2*nw) + float64(j)*(cellsize[w]/float64(nw))
			for α := 0; α < nx; α++ {
				rx := R[X] - cellsize[X]/2 + cellsize[X]/float64(2*nx) + (cellsize[X]/float64(nx))*float64(α)
				for β := 0; β < ny; β++ {
					ry := R[Y] - cellsize[Y]/2 + cellsize[Y]/float64(2*ny) + (cellsize[Y]/float64(ny))*float64(β)
					for γ := 0; γ < nz; γ++ {
						rz := R[Z] - cellsize[Z]/2 + cellsize[Z]/float64(2*nz) + (cellsize[Z]/float64(nz))*float64(γ)

						pole[u] = pu1
						R2[X], R2[Y], R2[Z] = rx-pole[X], ry-pole[Y], rz-pole[Z]
						r := math.Sqrt(R2[X]*R2[X] + R2[Y]*R2[Y] + R2[Z]*R2[Z])
						qr := charge / (4 * math.Pi * r * r * r)
						bx := R2[X] * qr
						by := R2[Y] * qr
						bz := R2[Z] * qr

						pole[u] = pu2
						R2[X], R2[Y], R2[Z] = rx-pole[X], ry-pole[Y], rz-pole[Z]
						r = math.Sqrt(R2[X]*R2[X] + R2[Y]*R2[Y] + R2[Z]*R2[Z])
						qr = -charge / (4 * math.Pi * r * r * r)
						B[X] += (bx + R2[X]*qr)
						B[Y] += (by + R2[Y]*qr)
						B[Z] += (bz + R2[Z]*qr)
					}
				}
			}
		}
	}
	return
}

// symmetrizeKernel reconstructs the skipped half spaces from symmetry and fills in the symmetric
// components. Copy of the tail of CalcDemagKernel, kept separate so that CalcDemagKernel stays
// untouched and can be deleted later without affecting the new methods.
func symmetrizeKernel(kernel [3][3]*data.Slice, array [3][3][][][]float32, size [3]int) [3][3]*data.Slice {
	// Reconstruct skipped parts from symmetry (X)
	for z := 0; z < size[Z]; z++ {
		for y := 0; y < size[Y]; y++ {
			for x := size[X]/2 + 1; x < size[X]; x++ {
				x2 := size[X] - x
				array[X][X][z][y][x] = array[X][X][z][y][x2]
				array[X][Y][z][y][x] = -array[X][Y][z][y][x2]
				array[X][Z][z][y][x] = -array[X][Z][z][y][x2]
				array[Y][Y][z][y][x] = array[Y][Y][z][y][x2]
				array[Y][Z][z][y][x] = array[Y][Z][z][y][x2]
				array[Z][Z][z][y][x] = array[Z][Z][z][y][x2]
			}
		}
	}

	// Reconstruct skipped parts from symmetry (Y)
	for z := 0; z < size[Z]; z++ {
		for y := size[Y]/2 + 1; y < size[Y]; y++ {
			y2 := size[Y] - y
			for x := 0; x < size[X]; x++ {
				array[X][X][z][y][x] = array[X][X][z][y2][x]
				array[X][Y][z][y][x] = -array[X][Y][z][y2][x]
				array[X][Z][z][y][x] = array[X][Z][z][y2][x]
				array[Y][Y][z][y][x] = array[Y][Y][z][y2][x]
				array[Y][Z][z][y][x] = -array[Y][Z][z][y2][x]
				array[Z][Z][z][y][x] = array[Z][Z][z][y2][x]

			}
		}
	}

	// Reconstruct skipped parts from symmetry (Z)
	for z := size[Z]/2 + 1; z < size[Z]; z++ {
		z2 := size[Z] - z
		for y := 0; y < size[Y]; y++ {
			for x := 0; x < size[X]; x++ {
				array[X][X][z][y][x] = array[X][X][z2][y][x]
				array[X][Y][z][y][x] = array[X][Y][z2][y][x]
				array[X][Z][z][y][x] = -array[X][Z][z2][y][x]
				array[Y][Y][z][y][x] = array[Y][Y][z2][y][x]
				array[Y][Z][z][y][x] = -array[Y][Z][z2][y][x]
				array[Z][Z][z][y][x] = array[Z][Z][z2][y][x]
			}
		}
	}

	// for 2D these elements are zero:
	if size[Z] == 1 {
		kernel[X][Z] = nil
		kernel[Y][Z] = nil
	}
	// make result symmetric for tools that expect it so.
	kernel[Y][X] = kernel[X][Y]
	kernel[Z][X] = kernel[X][Z]
	kernel[Z][Y] = kernel[Y][Z]
	return kernel
}
