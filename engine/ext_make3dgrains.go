// 3D Voronoi tessellation. Contributed by Peyton Murray.

package engine

import (
	"math"
	"math/rand"
)

var GrainCutShape = false // complete all voronoi grains whose centre lies within the shape. Warning, this also cuts away parts of the shape whose closest voronoi centre lies outside the shape

func init() {
	DeclFunc("ext_make3dgrains", Voronoi3d, "3D Voronoi tesselation over shape (grain size, starting region number, num regions, shape, seed)")
	DeclVar("ext_grainCutShape", &GrainCutShape, "Whether to add the complete (3D) voronoi grain, only if its centre lies within the shape (default=false)")
}

func Voronoi3d(grainsize float64, startRegion int, numRegions int, inputShape Shape, seed int) {
	Refer("Lel2014")
	SetBusy(true)
	defer SetBusy(false)

	t := newTesselation3d(grainsize, numRegions, int64(seed), startRegion, inputShape)
	regions.hist = append(regions.hist, t.RegionOf)
	regions.render(t.RegionOf)
}

type tesselation3d struct {
	grainsize   float64
	maxRegion   int
	rnd         *rand.Rand
	startRegion int
	shape       Shape      //Shape of the tesselated region
	centers     []center3d //List of Voronoi centers
}

// Stores location of each Voronoi center
type center3d struct {
	x, y, z float64 // center position (m)
	region  byte    // region for all cells near center
}

// Stores location of each cell
type cellLocs struct{ x, y, z float64 }

// nRegion exclusive
func newTesselation3d(grainsize float64, nRegion int, seed int64, startRegion int, inputShape Shape) *tesselation3d {
	t := tesselation3d{grainsize,
		nRegion,
		rand.New(rand.NewSource(seed)),
		startRegion,
		inputShape,
		make([]center3d, 0)}

	t.makeRandomCenters()
	return &t
}

// Permutes the slice of cell locations. I don't understand why this needs to be done if we're choosing
// random (Intn()) cells out of the slice of cell locations, but hey, it seems to do the trick.
func (t *tesselation3d) shuffleCells(src []cellLocs) []cellLocs {
	dest := make([]cellLocs, len(src))
	perm := t.rnd.Perm(len(src))
	for i, v := range perm {
		dest[v] = src[i]
	}
	return dest
}

func (t *tesselation3d) makeRandomCenters() {
	//Make a list of all the cells in the shape.
	cells := t.tabulateCells()
	cells = t.shuffleCells(cells)

	//Choose number of grains to make. Assume volume of grain is given by (4/3)*pi*r^3
	shapeVolume := cellVolume() * float64(len(cells))
	grainVolume := (float64(1) / 6) * math.Pi * t.grainsize * t.grainsize * t.grainsize
	nAvgGrains := shapeVolume / grainVolume
	nGrains := t.truncNorm(nAvgGrains)

	//TODO: same cell can be chosen twice by random chance
	t.centers = make([]center3d, nGrains)
	for p := 0; p < nGrains; p++ {
		rndCell := cells[t.rnd.Intn(nGrains)]
		t.centers[p].x = rndCell.x
		t.centers[p].y = rndCell.y
		t.centers[p].z = rndCell.z
		randRegion := t.startRegion + t.rnd.Intn(t.maxRegion)
		t.centers[p].region = byte(randRegion)
	}
}

// Creates a slice of all cells which fall in the shape specified in the constructor.
func (t *tesselation3d) tabulateCells() []cellLocs {
	//Initialze array of cells
	cells := make([]cellLocs, 0)

	//Get the mesh size
	meshSize := MeshSize()

	//Iterate across all cells in the mesh, and append those that are inside the shape
	for ix := 0; ix < meshSize[0]; ix++ {
		for iy := 0; iy < meshSize[1]; iy++ {
			for iz := 0; iz < meshSize[2]; iz++ {

				cell := Index2Coord(ix, iy, iz)

				x := cell.X()
				y := cell.Y()
				z := cell.Z()

				if t.shape(x, y, z) || GrainCutShape {
					cells = append(cells, cellLocs{x, y, z})
				}
			}
		}
	}

	print("Number of cells in region: ", len(cells), "\n")
	print("Number of cells in universe: ", meshSize[0]*meshSize[1]*meshSize[2], "\n")

	return cells
}

func (t *tesselation3d) RegionOf(x, y, z float64) int {
	// Check if the point is within the shape or if we're cutting the shape along the grains
	if !(t.shape(x, y, z) || GrainCutShape) {
		return -1 // Regions < 0 won't be rastered
	}

	// Find the nearest center point to the (x, y, z) position
	nearest := center3d{x, y, z, 0}
	mindist := math.Inf(1)
	for _, c := range t.centers {
		dist := sqr(x-c.x) + sqr(y-c.y) + sqr(z-c.z)
		if dist < mindist {
			nearest = c
			mindist = dist
		}
	}

	// Check if the nearest point's region should be returned
	if (t.shape(x, y, z) && !GrainCutShape) || (t.shape(nearest.x, nearest.y, nearest.z) && GrainCutShape) {
		return int(nearest.region)
	}

	return -1
}

// Generate normally distributed numbers; mean = lambda, variance = lambda. If generated number < 0, return 1.
// Equivalent to Poisson distribution (with  mean = lambda) for large lambda (which is usually true, since the volume
// of a grain is usually much less than the simulation volume.
func (t *tesselation3d) truncNorm(lambda float64) int {
	ret := lambda + math.Sqrt(lambda)*t.rnd.NormFloat64()
	if ret <= 0 {
		return 1
	} else {
		return int(ret + 0.5)
	}
}
