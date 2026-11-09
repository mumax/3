package mag

import "testing"

// Stray field kernel at zero offset must be equal to normal demag kernel.
// Since symmetries are not exploited, floating-point differences are not unexpected.
func TestZeroOffsetStrayIsDemag(t *testing.T) {
	inputSize := [3]int{4, 4, 1}
	pbc := [3]int{1, 1, 0}
	cellsize := [3]float64{1, 1, 1}
	size := padSize(inputSize, pbc)
	demag := CalcDemagKernel(inputSize, pbc, cellsize, 6)
	stray := CalcStrayKernel(inputSize, pbc, cellsize, 6, [3]float64{})

	var diffSq, demagSq float64
	for i := range 3 {
		for j := i; j < 3; j++ {
			if demag[i][j] == nil && stray[i][j] == nil {
				continue
			}
			for y := range size[Y] {
				for x := range size[X] {
					var want, got float32
					if demag[i][j] != nil {
						want = demag[i][j].Scalars()[0][y][x]
					}
					if stray[i][j] != nil {
						got = stray[i][j].Scalars()[0][y][x]
					}
					d := float64(got - want)
					diffSq += d * d
					v := float64(want)
					demagSq += v * v
				}
			}
		}
	}
	if relErrSq := diffSq / demagSq; relErrSq > 1e-10 {
		t.Fatalf("zero-offset stray kernel != demag kernel: relative squared error %g", relErrSq)
	}
}

// Offsets just beyond touching cells must not cause pathological quadrature
// subdivisions; this guards the integration-distance lower bound.
func TestCalcStrayKernelNearTouchingOffset(t *testing.T) {
	CalcStrayKernel(
		[3]int{2, 2, 1},
		[3]int{1, 1, 0},
		[3]float64{1, 1, 1},
		3,
		[3]float64{1 + 1e-6, 0, 0},
	)
}

// In 2D, contrary to demag, offset stray kernels still need out-of-plane cross terms.
func TestStrayKernel2DHasCrossTerms(t *testing.T) {
	inputSize := [3]int{1, 1, 1}
	pbc := [3]int{}
	cellsize := [3]float64{1, 1, 1}
	offset := [3]float64{0.25, 0, 1}
	cacheDir := t.TempDir()

	first := StrayKernel(inputSize, pbc, cellsize, 1, offset, cacheDir)
	if first[X][Z] == nil || first[Y][Z] == nil {
		t.Fatal("2D stray kernel is missing out-of-plane cross terms")
	}

	got := StrayKernel(inputSize, pbc, cellsize, 1, offset, cacheDir)
	if got[X][Z] == nil || got[Y][Z] == nil {
		t.Fatal("cached 2D stray kernel is missing out-of-plane cross terms")
	}
	if got[X][Z].Scalars()[0][0][0] != first[X][Z].Scalars()[0][0][0] {
		t.Fatal("cached stray kernel cross term differs from calculated kernel")
	}
}

// Under PBC, offsets differing by whole periods should describe the same
// physical observation point; this guards against image-range shifting.
func TestStrayKernelPBCOffsetIsPeriodic(t *testing.T) {
	inputSize := [3]int{4, 4, 1}
	pbc := [3]int{1, 1, 0}
	cellsize := [3]float64{1, 1, 1}
	offset := [3]float64{0.25, -0.5, 0}
	periodShiftedOffset := [3]float64{4.25, 3.5, 0}

	got := CalcStrayKernel(inputSize, pbc, cellsize, 3, offset)
	want := CalcStrayKernel(inputSize, pbc, cellsize, 3, periodShiftedOffset)
	for i := range 3 {
		for j := i; j < 3; j++ {
			gotValues := got[i][j].Scalars()[0]
			wantValues := want[i][j].Scalars()[0]
			for y := range gotValues {
				for x := range gotValues[y] {
					if gotValues[y][x] != wantValues[y][x] {
						t.Fatalf("kernel[%d][%d][%d][%d] differs for periodic offsets: got %g, want %g",
							i, j, y, x, gotValues[y][x], wantValues[y][x])
					}
				}
			}
		}
	}
}
