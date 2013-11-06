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

type Regions struct {
	arr        [][][]byte  // regions map: cell i,j,k -> byte index
	cpu        []byte      // arr data, stored contiguously
	gpuCache   *cuda.Bytes // gpu copy of cpu data, possibly out-of-sync
	gpuCacheOK bool        // gpuCache in sync with cpu
	maxreg     int         // highest used region
	doc
}

func (r *Regions) alloc() {
	mesh := r.Mesh()
	r.cpu = make([]byte, mesh.NCell())
	r.arr = resizeBytes(r.cpu, mesh.Size())
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
	if !ok {
		log.Println("Note: DefRegion ", id, ": shape is empty")
	}
}

func DefRegionCell(id int, x, y, z int) {
	defRegionId(id)
	regions.gpuCacheOK = false
	regions.arr[z][y][x] = byte(id)
}

func defRegionId(id int) {
	if id < 0 || id > NREGION {
		log.Fatalf("region id should be 0 -%v, have: %v", NREGION, id)
	}
	if id+1 > regions.maxreg {
		regions.maxreg = id + 1 // we loop < maxreg, so +1
	}
	checkMesh()
}

// normalized volume (0..1) of region.
// TODO: might be cached.
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

//func checkRegionIdx(id int) {
//	if id < 0 || id > NREGION {
//		log.Fatalf("region id should be 0-255, have: %v", id)
//	}
//}

// Set the region of one cell
func (r *Regions) SetCell(ix, iy, iz int, region int) {
	r.arr[iz][iy][ix] = byte(region)
	r.gpuCacheOK = false
}

// Get the region data on GPU, first uploading it if needed.
func (r *Regions) Gpu() *cuda.Bytes {
	if r.gpuCacheOK {
		return r.gpuCache
	}
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
	buf := cuda.Buffer(1, r.Mesh())
	cuda.RegionDecode(buf, unitMap.gpuLUT1(), regions.Gpu())
	return buf, true
}

// Re-interpret a contiguous array as a multi-dimensional array of given size.
func resizeBytes(array []byte, size [3]int) [][][]byte {
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

func (r *Regions) Mesh() *data.Mesh { return &globalmesh }
