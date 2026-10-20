package mag

import (
	"math"
	"math/big"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/mumax/3/data"
)

// maxAbsKernel returns max |N| over all components.
func maxAbsKernel(k [3][3]*data.Slice) float64 {
	m := 0.0
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			if k[i][j] != nil {
				for _, v := range k[i][j].Host()[0] {
					m = math.Max(m, math.Abs(float64(v)))
				}
			}
		}
	}
	return m
}

// compareKernels fails if any entry of got differs from ref by more than tol * max|ref| (over all
// components, the scale that matters for the field) and logs per-component differences.
func compareKernels(t *testing.T, name string, ref, got [3][3]*data.Slice, tol float64) {
	t.Helper()
	scale := maxAbsKernel(ref)
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			if (ref[i][j] == nil) != (got[i][j] == nil) {
				t.Fatalf("%s N%d%d: nil mismatch", name, i, j)
			}
			if ref[i][j] == nil {
				continue
			}
			a, b := ref[i][j].Host()[0], got[i][j].Host()[0]
			var maxD float64
			same := 0
			for k := range a {
				maxD = math.Max(maxD, math.Abs(float64(a[k]-b[k])))
				if a[k] == b[k] {
					same++
				}
			}
			t.Logf("%s N%d%d: max|diff| = %.1e of max|N|, bit-identical %d/%d", name, i, j, maxD/scale, same, len(a))
			if maxD > tol*scale {
				t.Errorf("%s N%d%d: max|diff| %.2e > %.0e of max|N|", name, i, j, maxD/scale, tol)
			}
		}
	}
}

// checkOddComponentsVanish: with Ny=1 and symmetric PBC in y, Nxy and Nyz are odd in y and must
// sum to zero. A float64 image sum gets this to rounding level; CalcDemagKernel's float32 sum
// leaves ~1e-9 of max|N|.
func checkOddComponentsVanish(t *testing.T, name string, k [3][3]*data.Slice) {
	t.Helper()
	scale := maxAbsKernel(k)
	for _, c := range [][2]int{{X, Y}, {Y, Z}} {
		if k[c[0]][c[1]] == nil {
			continue
		}
		m := 0.0
		for _, v := range k[c[0]][c[1]].Host()[0] {
			m = math.Max(m, math.Abs(float64(v)))
		}
		t.Logf("%s N%d%d (should be 0): max = %.1e of max|N|", name, c[0], c[1], m/scale)
		if m > 1e-10*scale {
			t.Errorf("%s N%d%d should vanish by symmetry: %.1e of max|N|", name, c[0], c[1], m/scale)
		}
	}
}

// Tolerance vs CalcDemagKernel: its float32 image sums are off by a few 1e-6 of max|N|, growing with
// the number of images per cell; the fast kernel sums in float64. Logic errors show up as >=1e-3.
const oldKernelTol = 1e-4

func TestFastKernelEquivalence(t *testing.T) {
	cases := []struct {
		name      string
		size, pbc [3]int
		cell      [3]float64
	}{
		{"thin film PBCy (scaled user case)", [3]int{128, 1, 8}, [3]int{3, 200, 0}, [3]float64{5e-9, 5e-9, 2e-9}},
		{"2D film PBCy", [3]int{64, 1, 1}, [3]int{0, 300, 0}, [3]float64{4e-9, 4e-9, 1e-9}},
		{"small film, huge PBCy", [3]int{8, 1, 2}, [3]int{2, 20000, 0}, [3]float64{5e-9, 5e-9, 2e-9}},
		{"3D PBC all", [3]int{8, 6, 5}, [3]int{2, 3, 1}, [3]float64{3e-9, 4e-9, 2e-9}},
		{"no PBC", [3]int{10, 7, 3}, [3]int{0, 0, 0}, [3]float64{2e-9, 3e-9, 5e-9}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t0 := time.Now()
			old := CalcDemagKernel(c.size, c.pbc, c.cell, 6)
			tOld := time.Since(t0)
			t0 = time.Now()
			fast, ok := CalcDemagKernelFast(c.size, c.pbc, c.cell, 6, 1<<30)
			tFast := time.Since(t0)
			if !ok {
				t.Fatal("fast kernel refused to run")
			}
			t.Logf("old %v, fast %v (%.1fx)", tOld, tFast, tOld.Seconds()/tFast.Seconds())
			compareKernels(t, "fast vs old", old, fast, oldKernelTol)
			if c.size[Y] == 1 && c.pbc[Y] > 0 {
				checkOddComponentsVanish(t, "fast", fast)
			}
		})
	}
}

// availableMemory sizes the image cache in real runs (the other tests pass a budget explicitly). If it
// returned 0, FasterPBCKernel = "CPU" would silently fall back to the original method for Ny=1 films.
func TestAvailableMemory(t *testing.T) {
	if m := availableMemory(); m <= 0 {
		t.Errorf("availableMemory() = %d on %s/%s", m, runtime.GOOS, runtime.GOARCH)
	} else {
		t.Logf("available memory: %.1f GiB", float64(m)/(1<<30))
	}
}

func TestFastKernelFallback(t *testing.T) {
	if _, ok := CalcDemagKernelFast([3]int{16, 1, 2}, [3]int{1, 10, 0}, [3]float64{1e-9, 1e-9, 1e-9}, 6, 0); ok {
		t.Fatal("expected fallback (ok=false) with zero memory budget")
	}
}

// Full user case (16384x1x32, PBC 3,1000,0), fast method only — the old one takes hours.
// Run with: MUMAX_BIG_KERNEL=1 go test -run BigKernel -v -timeout 0 ./mag
func TestBigKernel(t *testing.T) {
	if os.Getenv("MUMAX_BIG_KERNEL") == "" {
		t.Skip("set MUMAX_BIG_KERNEL=1")
	}
	t0 := time.Now()
	_, ok := CalcDemagKernelFast([3]int{16384, 1, 32}, [3]int{3, 1000, 0}, [3]float64{5e-9, 5e-9, 2e-9}, 6, availableMemory()/2)
	t.Logf("ok=%v, %v", ok, time.Since(t0))
}

func TestKernelMethod(t *testing.T) {
	for in, want := range map[string]string{"off": "old", "OLD": "old", "": "old", "CPU": "cpu", " cpu ": "cpu"} {
		if got, err := kernelMethod(in); err != nil || got != want {
			t.Errorf("kernelMethod(%q) = %q, %v; want %q", in, got, err, want)
		}
	}
	for _, bad := range []string{"true", "fast", "GPU", "auto"} {
		if _, err := kernelMethod(bad); err == nil {
			t.Errorf("kernelMethod(%q): expected error", bad)
		}
	}
}

// TestFastKernelExactSum: the two methods compute identical float64 per-image values (kernelB) and differ
// only in how they sum the PBC images. Against the exact (200-bit) sum of those values, every entry of the
// new kernel must be within one float32 rounding step; the original's float32 summation error is logged.
func TestFastKernelExactSum(t *testing.T) {
	inputSize, pbc, cs := [3]int{16, 1, 4}, [3]int{3, 2000, 0}, [3]float64{5e-9, 5e-9, 2e-9}
	old := CalcDemagKernel(inputSize, pbc, cs, 6)
	fast, ok := CalcDemagKernelFast(inputSize, pbc, cs, 6, 1<<30)
	if !ok {
		t.Fatal("fast kernel refused to run")
	}

	size := padSize(inputSize, pbc)
	r1, r2 := kernelRanges(size, pbc)
	L := math.Min(cs[X], math.Min(cs[Y], cs[Z]))
	first := func(t, c int) int { return r1[c] + ((t-r1[c])%size[c]+size[c])%size[c] }
	scale := maxAbsKernel(old)
	for s := 0; s < 3; s++ {
		var errOld, errNew [3]float64
		for zw := 0; zw <= size[Z]/2; zw++ {
			for yw := 0; yw <= size[Y]/2; yw++ {
				for xw := 0; xw <= size[X]/2; xw++ {
					var acc [3]*big.Float
					for d := range acc {
						acc[d] = new(big.Float).SetPrec(200)
					}
					for z := first(zw, Z); z <= r2[Z]; z += size[Z] {
						for y := first(yw, Y); y <= r2[Y]; y += size[Y] {
							for x := first(xw, X); x <= r2[X]; x += size[X] {
								B := kernelB(s, x, y, z, cs, 6, L)
								for d := s; d < 3; d++ {
									acc[d].Add(acc[d], new(big.Float).SetFloat64(B[d]))
								}
							}
						}
					}
					for d := s; d < 3; d++ {
						if old[s][d] == nil {
							continue
						}
						exact, _ := acc[d].Float64()
						n := float64(fast[s][d].Scalars()[zw][yw][xw])
						o := float64(old[s][d].Scalars()[zw][yw][xw])
						e32 := float64(float32(exact))
						ulp := math.Abs(float64(math.Nextafter32(float32(e32), float32(math.Inf(1)))) - e32)
						if math.Abs(n-exact) > math.Max(ulp, 1e-12*scale) {
							t.Errorf("N%d%d[%d,%d,%d]: new %g, exact %g: more than one float32 step apart", s, d, zw, yw, xw, n, exact)
						}
						errNew[d] = math.Max(errNew[d], math.Abs(n-exact))
						errOld[d] = math.Max(errOld[d], math.Abs(o-exact))
					}
				}
			}
		}
		for d := s; d < 3; d++ {
			if old[s][d] != nil {
				t.Logf("N%d%d error vs exact sum, of max|N|: original %.1e, new %.1e", s, d, errOld[d]/scale, errNew[d]/scale)
			}
		}
	}
}
