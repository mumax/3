package engine

import (
	"math"
	"math/rand"
	"testing"
)

// brute is the reference: explicit union of the copies at i·a1 + j·a2, i in [i1, i2), j in [j1, j2).
func brute(s Shape, a1x, a1y, a2x, a2y float64, i1, i2, j1, j2 int) Shape {
	return func(x, y, z float64) bool {
		for i := i1; i < i2; i++ {
			for j := j1; j < j2; j++ {
				fi, fj := float64(i), float64(j)
				if s(x-(fi*a1x+fj*a2x), y-(fi*a1y+fj*a2y), z) {
					return true
				}
			}
		}
		return false
	}
}

// compare checks got == ref at random points in the box |x|, |y| < half.
func compare(t *testing.T, name string, got, ref Shape, half float64, npts int) {
	t.Helper()
	rng := rand.New(rand.NewSource(1))
	in := 0
	for k := 0; k < npts; k++ {
		x, y, z := (rng.Float64()*2-1)*half, (rng.Float64()*2-1)*half, (rng.Float64()-0.5)*1e-8
		g, r := got(x, y, z), ref(x, y, z)
		if g != r {
			t.Fatalf("%s: got %v, want %v at (%g, %g, %g)", name, g, r, x, y, z)
		}
		if r {
			in++
		}
	}
	if in == 0 || in == npts {
		t.Fatalf("%s: test is vacuous (%d of %d points inside)", name, in, npts)
	}
}

const d = 50e-9 // honeycomb nearest-neighbour distance

var (
	hx1, hy1, hx2, hy2 = math.Sqrt(3) * d, 0.0, math.Sqrt(3) / 2 * d, 1.5 * d
	// two-part basis (honeycomb) plus a rotated square, and a big circle so neighbouring copies overlap
	basis = Circle(20e-9).Transl(0, d/2, 0).Add(Square(30e-9).RotZ(0.3).Transl(0, -d/2, 0))
	big   = Ellipse(120e-9, 40e-9).Transl(10e-9, 5e-9, 0)
)

func TestLattice2DFinite(t *testing.T) {
	for _, s := range []Shape{basis, big} {
		for _, n := range [][2]int{{2, 2}, {4, 3}, {5, 5}, {7, 2}, {2, 6}} {
			got := s.Lattice2D(hx1, hy1, hx2, hy2, n[0], n[1])
			compare(t, "finite", got, brute(s, hx1, hy1, hx2, hy2, 0, n[0], 0, n[1]), 600e-9, 100000)
		}
	}
	// a non-reduced oblique basis (angle 150°)
	got := basis.Lattice2D(90e-9, 0, -80e-9, 45e-9, 4, 5)
	compare(t, "oblique", got, brute(basis, 90e-9, 0, -80e-9, 45e-9, 0, 4, 0, 5), 600e-9, 100000)
	// rotated lattice: all four vector components non-zero
	for _, s := range []Shape{basis, big} {
		got = s.Lattice2D(80e-9, 35e-9, -30e-9, 90e-9, 5, 4)
		compare(t, "rotated", got, brute(s, 80e-9, 35e-9, -30e-9, 90e-9, 0, 5, 0, 4), 600e-9, 100000)
		got = s.Lattice2D(80e-9, 35e-9, -30e-9, 90e-9, 0, 0)
		compare(t, "rotated inf", got, brute(s, 80e-9, 35e-9, -30e-9, 90e-9, -15, 15, -15, 15), 400e-9, 20000)
	}
}

// n = 0 is infinite; compared with a finite lattice much larger than the test box.
func TestLattice2DInfinite(t *testing.T) {
	const N = 15
	for _, s := range []Shape{basis, big} {
		compare(t, "inf×inf", s.Lattice2D(hx1, hy1, hx2, hy2, 0, 0), brute(s, hx1, hy1, hx2, hy2, -N, N, -N, N), 400e-9, 20000)
		compare(t, "inf×3", s.Lattice2D(hx1, hy1, hx2, hy2, 0, 3), brute(s, hx1, hy1, hx2, hy2, -N, N, 0, 3), 400e-9, 20000)
		compare(t, "4×inf", s.Lattice2D(hx1, hy1, hx2, hy2, 4, 0), brute(s, hx1, hy1, hx2, hy2, 0, 4, -N, N), 400e-9, 20000)
	}
}

func TestLattice1D(t *testing.T) {
	const ax, ay = 60e-9, 25e-9
	for _, s := range []Shape{basis, big, Rect(20e-9, 400e-9)} {
		for _, n := range []int{2, 3, 6} {
			compare(t, "1D", s.Lattice1D(ax, ay, n), brute(s, ax, ay, 0, 0, 0, n, 0, 1), 500e-9, 50000)
		}
		compare(t, "1D inf", s.Lattice1D(ax, ay, 0), brute(s, ax, ay, 0, 0, -30, 30, 0, 1), 500e-9, 50000)
	}
}

// n counts the copies: n = 1 is the shape itself, n = 2 adds one copy at +a.
func TestLatticeCount(t *testing.T) {
	s := Circle(10e-9)
	l := s.Lattice2D(30e-9, 0, 0, 40e-9, 3, 2)
	for i := -1; i <= 3; i++ {
		for j := -1; j <= 2; j++ {
			want := i >= 0 && i < 3 && j >= 0 && j < 2
			if l(float64(i)*30e-9, float64(j)*40e-9, 0) != want {
				t.Errorf("copy (%d,%d): got %v, want %v", i, j, !want, want)
			}
		}
	}
	one := s.Lattice1D(30e-9, 0, 2)
	if !one(0, 0, 0) || !one(30e-9, 0, 0) || one(-30e-9, 0, 0) || one(60e-9, 0, 0) {
		t.Error("Lattice1D(n=2) is not the shape plus one copy at +a")
	}
}

// Degenerate input is routed to the 1D or 0D case instead of inverting a singular matrix.
func TestLatticeDegenerate(t *testing.T) {
	s := basis
	same := func(name string, a, b Shape) { compare(t, name, a, b, 600e-9, 50000) }
	for _, n2 := range []int{0, 1, 5} {
		same("a2=0", s.Lattice2D(hx1, hy1, 0, 0, 4, n2), s.Lattice1D(hx1, hy1, 4))
		same("a1=0", s.Lattice2D(0, 0, hx2, hy2, n2, 4), s.Lattice1D(hx2, hy2, 4))
	}
	same("n2=1", s.Lattice2D(hx1, hy1, hx2, hy2, 0, 1), s.Lattice1D(hx1, hy1, 0))
	same("n1=1", s.Lattice2D(hx1, hy1, hx2, hy2, 1, 3), s.Lattice1D(hx2, hy2, 3))
	for _, l := range []Shape{
		s.Lattice2D(0, 0, 0, 0, 0, 0), s.Lattice2D(hx1, hy1, hx2, hy2, 1, 1),
		s.Lattice1D(0, 0, 0), s.Lattice1D(0, 0, 7), s.Lattice1D(hx1, hy1, 1),
	} {
		same("no repetition", l, s)
	}
}

func TestLatticeInvalid(t *testing.T) {
	for name, f := range map[string]func(){
		"n1<0":     func() { Circle(1).Lattice2D(1, 0, 0, 1, -1, 3) },
		"n2<0":     func() { Circle(1).Lattice2D(1, 0, 0, 1, 3, -2) },
		"n<0 (1D)": func() { Circle(1).Lattice1D(1, 0, -1) },
		"parallel": func() { Circle(1).Lattice2D(1, 2, -2, -4, 3, 3) },
		"NaN":      func() { Circle(1).Lattice2D(math.NaN(), 0, 0, 1, 3, 3) },
		"Inf":      func() { Circle(1).Lattice1D(math.Inf(1), 0, 0) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: no error", name)
				}
			}()
			f()
		}()
	}
	// parallel vectors are fine when one direction is not repeated
	Circle(1).Lattice2D(1, 2, -2, -4, 3, 1)
}

// The z dependence of a 3D shape is kept (lattice vectors are in-plane).
func TestLatticeKeepsZ(t *testing.T) {
	s := Cylinder(20e-9, 10e-9).Lattice2D(40e-9, 0, 0, 40e-9, 0, 0)
	if !s(40e-9, -80e-9, 4e-9) || s(40e-9, -80e-9, 6e-9) {
		t.Error("z extent of Cylinder not preserved")
	}
}

// The methods are reachable from mx3 scripts (case-insensitive, int arguments).
func TestLatticeScript(t *testing.T) {
	s, ok := World.MustEval("Circle(20e-9).Transl(0, 25e-9, 0).Add(Square(30e-9).RotZ(0.3).Transl(0, -25e-9, 0)).Lattice2D(sqrt(3)*50e-9, 0, sqrt(3)/2*50e-9, 1.5*50e-9, 5, 0)").(Shape)
	if !ok {
		t.Fatal("Lattice2D did not return a Shape")
	}
	compare(t, "script", s, basis.Lattice2D(hx1, hy1, hx2, hy2, 5, 0), 600e-9, 20000)
	if _, ok := World.MustEval("square(10e-9).lattice1d(30e-9, 0, 3)").(Shape); !ok {
		t.Fatal("lattice1d did not return a Shape")
	}
}

// bruteSym is the reference for SymmetricLattice2D: explicit union over |i|, |j| ≤ N with keep(i, j).
func bruteSym(s Shape, a1x, a1y, a2x, a2y float64, N int, keep func(i, j float64) bool) Shape {
	return func(x, y, z float64) bool {
		for i := -N; i <= N; i++ {
			for j := -N; j <= N; j++ {
				fi, fj := float64(i), float64(j)
				if keep(fi, fj) && s(x-(fi*a1x+fj*a2x), y-(fi*a1y+fj*a2y), z) {
					return true
				}
			}
		}
		return false
	}
}

var metrics = map[string]func(r float64) func(i, j float64) bool{
	"chebyshev": func(r float64) func(i, j float64) bool {
		return func(i, j float64) bool { return math.Max(math.Abs(i), math.Abs(j)) <= r }
	},
	"taxicab": func(r float64) func(i, j float64) bool {
		return func(i, j float64) bool { return math.Abs(i)+math.Abs(j) <= r }
	},
	"euclidean": func(r float64) func(i, j float64) bool {
		return func(i, j float64) bool { return math.Sqrt(i*i+j*j) <= r }
	},
}

// Number of shapes: n = 1 is the shape, n = 2 gives 9 (chebyshev) / 5 (taxicab) / 5 (euclidean), ...
func TestSymmetricLatticeCount(t *testing.T) {
	const a = 100e-9
	want := map[string]map[float64]int{
		"chebyshev": {1: 1, 1.9: 1, 2: 9, 2.5: 9, 3: 25, 4: 49},
		"taxicab":   {1: 1, 1.9: 1, 2: 5, 2.5: 5, 3: 13, 4: 25},
		"euclidean": {1: 1, 1.9: 1, 2: 5, 2.5: 9, 3: 13, 3.3: 21, 4: 29},
	}
	for metric, cases := range want {
		for n, count := range cases {
			l := Circle(10e-9).SymmetricLattice2D(a, 0, 0, a, n, metric)
			got := 0
			for i := -6; i <= 6; i++ {
				for j := -6; j <= 6; j++ {
					if l(float64(i)*a, float64(j)*a, 0) {
						got++
						if !metrics[metric](n-1)(float64(i), float64(j)) {
							t.Errorf("%s n=%g: unexpected copy (%d,%d)", metric, n, i, j)
						}
					}
				}
			}
			if got != count {
				t.Errorf("%s n=%g: %d shapes, want %d", metric, n, got, count)
			}
		}
	}
}

// Equal to the explicit union, with overlapping copies, honeycomb and rotated vectors.
func TestSymmetricLatticeUnion(t *testing.T) {
	vecs := [][4]float64{{hx1, hy1, hx2, hy2}, {80e-9, 35e-9, -30e-9, 90e-9}}
	for metric, keep := range metrics {
		for _, v := range vecs {
			for _, s := range []Shape{basis, big} {
				for _, n := range []float64{2, 3, 3.7, 5} {
					got := s.SymmetricLattice2D(v[0], v[1], v[2], v[3], n, metric)
					compare(t, metric, got, bruteSym(s, v[0], v[1], v[2], v[3], 6, keep(n-1)), 600e-9, 50000)
				}
			}
		}
	}
}

func TestSymmetricLatticeSpecial(t *testing.T) {
	s := basis
	same := func(name string, a, b Shape) { compare(t, name, a, b, 600e-9, 30000) }
	for metric := range metrics {
		// fractional n is floor(n) for chebyshev and taxicab
		if metric != "euclidean" {
			same("floor", s.SymmetricLattice2D(hx1, hy1, hx2, hy2, 3.9, metric), s.SymmetricLattice2D(hx1, hy1, hx2, hy2, 3, metric))
		}
		// infinite
		inf := s.Lattice2D(hx1, hy1, hx2, hy2, 0, 0)
		same("n=0", s.SymmetricLattice2D(hx1, hy1, hx2, hy2, 0, metric), inf)
		same("n=inf", s.SymmetricLattice2D(hx1, hy1, hx2, hy2, math.Inf(1), metric), inf)
		// zero vector: symmetric chain with 2n-1 copies, the same for every metric
		chain := s.Lattice1D(hx2, hy2, 5).Transl(-2*hx2, -2*hy2, 0)
		same("a1=0", s.SymmetricLattice2D(0, 0, hx2, hy2, 3, metric), chain)
		same("a2=0", s.SymmetricLattice2D(hx2, hy2, 0, 0, 3.5, metric), chain)
		same("a1=a2=0", s.SymmetricLattice2D(0, 0, 0, 0, 4, metric), s)
		// one layer: the shape itself, even for parallel vectors
		same("n=1", s.SymmetricLattice2D(1e-9, 2e-9, 2e-9, 4e-9, 1, metric), s)
	}
	same("hexagonal a2=0", s.SymmetricLattice2D(hx2, hy2, 0, 0, 3, "hexagonal"), s.Lattice1D(hx2, hy2, 5).Transl(-2*hx2, -2*hy2, 0))
	same("case-insensitive", s.SymmetricLattice2D(hx1, hy1, hx2, hy2, 3, "TaxiCab"), s.SymmetricLattice2D(hx1, hy1, hx2, hy2, 3, "taxicab"))
}

func TestSymmetricLatticeInvalid(t *testing.T) {
	for name, f := range map[string]func(){
		"n=0.5":    func() { Circle(1).SymmetricLattice2D(1, 0, 0, 1, 0.5, "taxicab") },
		"n<0":      func() { Circle(1).SymmetricLattice2D(1, 0, 0, 1, -2, "taxicab") },
		"n=NaN":    func() { Circle(1).SymmetricLattice2D(1, 0, 0, 1, math.NaN(), "taxicab") },
		"metric":   func() { Circle(1).SymmetricLattice2D(1, 0, 0, 1, 2, "manhattan") },
		"parallel": func() { Circle(1).SymmetricLattice2D(1, 2, -2, -4, 2, "euclidean") },
		"vec Inf":  func() { Circle(1).SymmetricLattice2D(math.Inf(1), 0, 0, 1, 2, "euclidean") },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: no error", name)
				}
			}()
			f()
		}()
	}
}

func TestSymmetricLatticeScript(t *testing.T) {
	for _, src := range []string{
		`circle(10e-9).symmetricLattice2D(100e-9, 0, 50e-9, sqrt(3)/2*100e-9, 2.5, "euclidean")`,
		`circle(10e-9).SymmetricLattice2D(100e-9, 0, 50e-9, sqrt(3)/2*100e-9, 3, "Chebyshev")`,
		`circle(10e-9).SymmetricLattice2D(100e-9, 0, 50e-9, sqrt(3)/2*100e-9, 4, "hexagonal")`,
	} {
		if _, ok := World.MustEval(src).(Shape); !ok {
			t.Fatalf("%s did not return a Shape", src)
		}
	}
}

// hexDist: lattice distance of each site (i, j) as the number of nearest-neighbour steps,
// found by breadth-first search over the shortest real-space lattice vectors (independent
// of the cube-coordinate formula in the code).
func hexDist(a1x, a1y, a2x, a2y float64, maxd int) map[[2]int]int {
	length := func(i, j int) float64 {
		return math.Hypot(float64(i)*a1x+float64(j)*a2x, float64(i)*a1y+float64(j)*a2y)
	}
	var nb [][2]int
	min := math.Inf(1)
	for i := -2; i <= 2; i++ {
		for j := -2; j <= 2; j++ {
			if (i != 0 || j != 0) && length(i, j) < min {
				min = length(i, j)
			}
		}
	}
	for i := -2; i <= 2; i++ {
		for j := -2; j <= 2; j++ {
			if (i != 0 || j != 0) && length(i, j) < min*(1+1e-9) {
				nb = append(nb, [2]int{i, j})
			}
		}
	}
	dist := map[[2]int]int{{0, 0}: 0}
	front := [][2]int{{0, 0}}
	for d := 1; d <= maxd; d++ {
		var next [][2]int
		for _, p := range front {
			for _, o := range nb {
				q := [2]int{p[0] + o[0], p[1] + o[1]}
				if _, seen := dist[q]; !seen {
					dist[q] = d
					next = append(next, q)
				}
			}
		}
		front = next
	}
	return dist
}

// Hexagonal metric on a hexagonal lattice: hexagonal numbers of shapes, a perfect hexagon
// (invariant under 60° rotation), the same set for the 60° and the 120° choice of a2.
func TestSymmetricLatticeHexagonal(t *testing.T) {
	const a = 100e-9
	a2_60 := [2]float64{a / 2, math.Sqrt(3) / 2 * a}
	a2_120 := [2]float64{-a / 2, math.Sqrt(3) / 2 * a}
	sites := func(l Shape) map[[2]int]bool { // sites by integer index in the 60° basis
		m := map[[2]int]bool{}
		for i := -8; i <= 8; i++ {
			for j := -8; j <= 8; j++ {
				if l(float64(i)*a+float64(j)*a2_60[0], float64(j)*a2_60[1], 0) {
					m[[2]int{i, j}] = true
				}
			}
		}
		return m
	}
	for n, count := range map[float64]int{1: 1, 2: 7, 2.9: 7, 3: 19, 4: 37, 5: 61} {
		l60 := Circle(10e-9).SymmetricLattice2D(a, 0, a2_60[0], a2_60[1], n, "hexagonal")
		l120 := Circle(10e-9).SymmetricLattice2D(a, 0, a2_120[0], a2_120[1], n, "Hexagonal")
		s60, s120 := sites(l60), sites(l120)
		if len(s60) != count || len(s120) != count {
			t.Errorf("n=%g: %d (60°) and %d (120°) shapes, want %d", n, len(s60), len(s120), count)
		}
		for p := range s60 {
			if !s120[p] {
				t.Errorf("n=%g: site %v only with the 60° basis", n, p)
			}
			// rotation by 60° in the 60° basis: (i, j) -> (-j, i+j)
			if !s60[[2]int{-p[1], p[0] + p[1]}] {
				t.Errorf("n=%g: not 60°-symmetric at %v", n, p)
			}
		}
	}
	// union with overlapping shapes vs brute force, honeycomb vectors (60°) and their 120° form
	for _, v := range [][4]float64{{hx1, hy1, hx2, hy2}, {hx1, hy1, hx2 - hx1, hy2 - hy1}} {
		dist := hexDist(v[0], v[1], v[2], v[3], 10)
		for _, s := range []Shape{basis, big} {
			for _, n := range []float64{2, 3, 4.5} {
				keep := func(i, j float64) bool {
					d, ok := dist[[2]int{int(i), int(j)}]
					return ok && float64(d) <= n-1
				}
				got := s.SymmetricLattice2D(v[0], v[1], v[2], v[3], n, "hexagonal")
				compare(t, "hexagonal", got, bruteSym(s, v[0], v[1], v[2], v[3], 6, keep), 600e-9, 50000)
			}
		}
	}
}
