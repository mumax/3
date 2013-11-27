package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"log"
)

var regions = Regions{doc: Doc(1, "regions", "")} // global regions map

const NREGION = 256 // maximum number of regions. (!) duplicated in CUDA

func init() {
	DeclFunc("DefRegion", DefRegion, "Define a material region with given index (0-255) and shape")
	DeclFunc("DefRegionCell", DefRegionCell, "Set a material region in one cell by index")
	DeclROnly("regions", &regions, "Outputs the region index for each cell")
}

// stores the region index for each cell
type Regions struct {
	arr        [][][]byte  // regions map: cell i,j,k -> byte index
	cpu        []byte      // arr data, stored contiguously
	gpuCache   *cuda.Bytes // gpu copy of cpu data, possibly out-of-sync
	gpuCacheOK bool        // gpuCache in sync with cpu
	deflist    []struct {
		r int
		s Shape
	} // history
	doc
}

func (r *Regions) alloc() {
	mesh := r.Mesh()
	r.cpu = make([]byte, mesh.NCell())
	r.arr = reshapeBytes(r.cpu, mesh.Size())
	r.gpuCache = cuda.NewBytes(mesh)
	DefRegion(0, universe)
}

// Define a region with id (0-255) to be inside the Shape.
func DefRegion(id int, s Shape) {
	defRegionId(id)
	regions.gpuCacheOK = false

	ok := false
	n := Mesh().Size()

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				r := Index2Coord(ix, iy, iz)
				if s(r[X], r[Y], r[Z]) { // inside
					regions.arr[iz][iy][ix] = byte(id)
					ok = true
				}
			}
		}
	}

	regions.deflist = append(regions.deflist, struct {
		r int
		s Shape
	}{id, s})

	if !ok {
		util.Log("Note: DefRegion ", id, ": shape is empty")
	}
}

func DefRegionCell(id int, x, y, z int) {
	defRegionId(id)
	regions.gpuCacheOK = false
	regions.arr[z][y][x] = byte(id)
}

func defRegionId(id int) {
	if id < 0 || id > NREGION {
		util.Fatalf("region id should be 0 -%v, have: %v", NREGION, id)
	}
	checkMesh()
}

// normalized volume (0..1) of region.
func (r *Regions) volume(region int) float64 {
	vol := 0
	reg := byte(region)
	for _, c := range r.cpu {
		if c == reg {
			vol++
		}
	}
	return float64(vol) / float64(globalmesh.NCell())
}

// Set the region of one cell
func (r *Regions) SetCell(ix, iy, iz int, region int) {
	// TODO: broken if also shifting! rm cpucache alltogehter, use local cache in loop
	r.arr[iz][iy][ix] = byte(region)
	r.gpuCacheOK = false
}

// Get the region data on GPU, first uploading it if needed.
func (r *Regions) Gpu() *cuda.Bytes {
	if r.gpuCacheOK {
		return r.gpuCache
	}
	log.Print("regions.upload")
	r.gpuCache.Upload(r.cpu)
	r.gpuCacheOK = true
	return r.gpuCache
}

var unitMap inputParam // unit map used to output regions quantity

func init() {
	unitMap.init(1, "unit", "", nil)
	for r := 0; r < NREGION; r++ {
		unitMap.setRegion(r, []float64{float64(r)})
	}
}

// Get returns the regions as a slice of floats, so it can be output.
func (r *Regions) Slice() (*data.Slice, bool) {
	buf := cuda.Buffer(1, r.Mesh().Size())
	cuda.RegionDecode(buf, unitMap.gpuLUT1(), regions.Gpu())
	return buf, true
}

// Re-interpret a contiguous array as a multi-dimensional array of given size.
func reshapeBytes(array []byte, size [3]int) [][][]byte {
	Nx, Ny, Nz := size[X], size[Y], size[Z]
	util.Argument(Nx*Ny*Nz == len(array))
	sliced := make([][][]byte, Nz)
	for i := range sliced {
		sliced[i] = make([][]byte, Ny)
	}
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = array[(i*Ny+j)*Nx+0 : (i*Ny+j)*Nx+Nx]
		}
	}
	return sliced
}

func (b *Regions) shift(dx int) {
	// TODO: return if no regions defined
	log.Println("regionshift", dx)
	r1 := b.Gpu()
	r2 := cuda.NewBytes(b.Mesh()) // TODO: somehow recycle
	defer r2.Free()
	newreg := byte(0) // new region at edge
	cuda.ShiftBytes(r2, r1, b.Mesh(), dx, newreg)
	r1.Copy(r2)

	// TODO: dedup from geom.shift!

	//n := b.Mesh().Size()
	//nx := n[X]
	// re-evaluate edge regions
	//var x1, x2 int
	//util.Argument(dx != 0)
	//if dx < 0 {
	//	x1 = nx + dx
	//	x2 = nx
	//} else {
	//	x1 = 0
	//	x2 = dx
	//}

	//panic("todo")
	//	for iz := 0; iz < n[Z]; iz++ {
	//		for iy := 0; iy < n[Y]; iy++ {
	//			for ix := x1; ix < x2; ix++ {
	//				r := Index2Coord(ix, iy, iz) // includes shift
	//				reg :=
	//				if !g.shape(r[X], r[Y], r[Z]) {
	//					g.SetCell(ix, iy, iz, 0) // a bit slowish, but hardly reached
	//				}
	//			}
	//		}
	//	}
	//
}

func (r *Regions) Mesh() *data.Mesh { return &globalmesh }
