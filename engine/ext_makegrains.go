package engine

import (
	"math"
	"math/rand"
)

func init() {
	DeclFunc("ext_makegrains", Voronoi, "Voronoi tesselation (grain size, num regions)")
}

func Voronoi(grainsize float64, numRegions, seed int) {
	Refer("Lel2014")
	SetBusy(true)
	defer SetBusy(false)

	t := newTesselation(grainsize, numRegions, int64(seed))
	regions.hist = append(regions.hist, t.RegionOf)
	regions.render(t.RegionOf)
}

type tesselation struct {
	grainsize float64
	tilesize  float64
	maxRegion int
	cache     map[int2][]center
	seed      int64
	rnd       *rand.Rand
}

// integer tile coordinate
type int2 struct{ x, y int }

// Voronoi center info
type center struct {
	x, y   float64 // center position (m)
	region byte    // region for all cells near center
}

// nRegion exclusive
func newTesselation(grainsize float64, nRegion int, seed int64) *tesselation {
	return &tesselation{grainsize,
		float64(float32(grainsize * TILE)), // expect 4 grains/block, 36 per 3x3 blocks = safe, relatively round number
		nRegion,
		make(map[int2][]center),
		seed,
		rand.New(rand.NewSource(0))}
}

const (
	TILE   = 2           // tile size in grains
	LAMBDA = TILE * TILE // expected grains per tile
)

// Returns the region of the grain where cell at x,y,z belongs to
func (t *tesselation) RegionOf(x, y, z float64) int {
	tile := t.tileOf(x, y) // tile containing x,y

	// look for nearest center in tile + neighbors
	nearest := center{x, y, 0} // dummy initial value, but safe should the infinite impossibility strike.
	mindist := math.Inf(1)
	for tx := tile.x - 1; tx <= tile.x+1; tx++ {
		for ty := tile.y - 1; ty <= tile.y+1; ty++ {
			centers := t.centersInTile(tx, ty)
			for _, c := range centers {
				dist := sqr(x-c.x) + sqr(y-c.y)
				if dist < mindist {
					nearest = c
					mindist = dist
				}
			}
		}
	}

	//fmt.Println("nearest", x, y, ":", nearest)
	return int(nearest.region)
}

// Returns the list of Voronoi centers in tile(ix, iy), using only ix,iy to seed the random generator
func (t *tesselation) centersInTile(tx, ty int) []center {
	pos := int2{tx, ty}
	if c, ok := t.cache[pos]; ok {
		return c
	} else {
		// tile-specific seed that works for positive and negative tx, ty
		seed := (int64(ty)+(1<<24))*(1<<24) + (int64(tx) + (1 << 24))
		t.rnd.Seed(seed ^ t.seed)
		N := t.poisson(LAMBDA)
		c := make([]center, N)

		// absolute position of tile (m)
		x0, y0 := float64(tx)*t.tilesize, float64(ty)*t.tilesize

		for i := range c {
			// random position inside tile
			c[i].x = x0 + t.rnd.Float64()*t.tilesize
			c[i].y = y0 + t.rnd.Float64()*t.tilesize
			c[i].region = byte(t.rnd.Intn(t.maxRegion))
		}
		t.cache[pos] = c
		return c
	}
}

func sqr(x float64) float64 { return x * x }

func (t *tesselation) tileOf(x, y float64) int2 {
	ix := int(math.Floor(x / t.tilesize))
	iy := int(math.Floor(y / t.tilesize))
	return int2{ix, iy}
}

// Generate poisson distributed numbers (according to Knuth)
func (t *tesselation) poisson(lambda float64) int {
	L := math.Exp(-lambda)
	k := 1
	p := t.rnd.Float64()
	for p > L {
		k++
		p *= t.rnd.Float64()
	}
	return k - 1
}
