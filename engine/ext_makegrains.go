package engine

import (
	"fmt"
	"math"
	"math/rand"
)

func init() {
	DeclFunc("ext_makegrains", Voronoi, "Voronoi tesselation (grain size, num regions)")
}

func Voronoi(grainsize float64, numRegions, seed int) {
	t := newTesselation(grainsize, numRegions, int64(seed))

	r := &regions
	n := Mesh().Size()
	l := r.HostList() // need to start from previous state
	arr := reshapeBytes(l, r.Mesh().Size())

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				r := Index2Coord(ix, iy, iz)
				arr[iz][iy][ix] = t.RegionOf(r[X], r[Y], r[Z])
			}
		}
	}
	fmt.Println(l)
	r.gpuCache.Upload(l)
}

type tesselation struct {
	grainsize float64
	tilesize  float64
	maxRegion int
	cache     map[int2][]center
	rnd       *rand.Rand
}

// integer tile coordinate
type int2 struct{ x, y int }

// Voronoi center info
type center struct {
	x, y   float64 // center position (m)
	region byte    // region for all cells near center
}

func newTesselation(grainsize float64, maxRegion int, seed int64) *tesselation {
	return &tesselation{grainsize,
		float64(float32(grainsize * TILE)), // expect 4 grains/block, 36 per 3x3 blocks = safe, relatively round number
		maxRegion,
		make(map[int2][]center),
		rand.New(rand.NewSource(0))}
}

const (
	TILE   = 2           // tile size in grains
	LAMBDA = TILE * TILE // expected grains per tile
)

// Returns the region of the grain where cell at x,y,z belongs to
func (t *tesselation) RegionOf(x, y, z float64) byte {
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
	return nearest.region
}

// Returns the list of Voronoi centers in tile(ix, iy), using only ix,iy to seed the random generator
func (t *tesselation) centersInTile(tx, ty int) []center {
	pos := int2{tx, ty}
	if c, ok := t.cache[pos]; ok {
		return c
	} else {
		// tile-specific seed that works for positive and negative tx, ty
		seed := ((int64(ty) + (2 << 30)) << 30) + (int64(tx) + (2 << 30))
		t.rnd.Seed(seed)
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
	ix := int(x / t.tilesize)
	iy := int(y / t.tilesize)
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

//
//func makeGrains(startregion, stopRegion, numberofgrains int) {
//	panic("TODO")
//	//	numberofregions := stopRegion - startregion
//	//	//log.Printf("making grains in %v regions, starting from %v, number of Voronoi centra=%v", numberofregions, startregion, numberofgrains)
//	//	util.Argument(len(regions.arr) == 1) // 2D only
//	//
//	//	for i := startregion; i < startregion+numberofregions; i++ {
//	//		DefRegion(i, func(x, y, z float64) bool { return false })
//	//	}
//	//	arr := regions.arr[0]
//	//
//	//	//make voronoi:
//	//	voronoi := make([][]int, len(arr))
//	//	for i := range voronoi {
//	//		voronoi[i] = make([]int, len(arr[i]))
//	//	}
//	//
//	//	// choose "numberofgrains" cells random in whole geometry en and give them voronoinumber
//	//	//	store the voronoicenters in different slice, to be used in next step
//	//	voronoicentra := make([][]int, numberofgrains)
//	//	for i := range voronoicentra {
//	//		voronoicentra[i] = make([]int, 2)
//	//	}
//	//
//	//	for voronoinumber := 0; voronoinumber < numberofgrains; voronoinumber++ {
//	//		j := rand.Intn(len(arr))
//	//		k := rand.Intn(len(arr[j]))
//	//		voronoicentra[voronoinumber][0] = j
//	//		voronoicentra[voronoinumber][1] = k
//	//		voronoi[j][k] = voronoinumber
//	//	}
//	//
//	//	//2: Find for every cell the closest voronoicentre and change its voronoinumber to the number of the centre
//	//
//	//	for j := 0; j < len(arr); j++ {
//	//		for k := 0; k < len(arr[j]); k++ {
//	//			distance := float64(math.Sqrt(math.Pow(float64(j-voronoicentra[0][0]), 2.) + math.Pow(float64(k-voronoicentra[0][1]), 2.)))
//	//			newdist := float64(0)
//	//			for i := range voronoicentra {
//	//				newdist = float64(math.Sqrt(math.Pow(float64(j-voronoicentra[i][0]), 2.) + math.Pow(float64(k-voronoicentra[i][1]), 2.)))
//	//				if newdist < distance {
//	//					voronoi[j][k] = i
//	//					distance = newdist
//	//				}
//	//			}
//	//		}
//	//	}
//	//
//	//	//3: give every cell its region by diividing its voronoinumber modulo number of regions
//	//	A := regions.arr
//	//	for _, arr := range A {
//	//		for j := 0; j < len(arr); j++ {
//	//			for k := 0; k < len(arr[j]); k++ {
//	//				arr[j][k] = byte(startregion + voronoi[j][k]%numberofregions)
//	//			}
//	//		}
//	//	}
//}
