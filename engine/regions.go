package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"log"
)

var regions = Regions{doc: Doc(1, "regions", "")} // global regions map

const NREGION = 256 // maximum number of regions. (!) duplicated in CUDA

func init() {
	DeclFunc("DefRegion", DefRegion, "Define a material region with given index (0-255) and shape")
	DeclROnly("regions", &regions, "Outputs the region index for each cell")
}

type Regions struct {
	arr        [][][]byte  // regions map: cell i,j,k -> byte index
	cpu        []byte      // arr data, stored contiguously
	gpuCache   *cuda.Bytes // gpu copy of cpu data, possibly out-of-sync
	gpuCacheOK bool        // gpuCache in sync with cpu
	maxreg     int         // highest used region
	//defined    [MAXREG]bool // has region i been defined already (not allowed to set it if not defined)
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
	if id < 0 || id > NREGION {
		log.Fatalf("region id should be 0 -%v, have: %v", NREGION, id)
	}
	if id+1 > regions.maxreg {
		regions.maxreg = id + 1 // we loop < maxreg, so +1
	}
	//regions.defined[id] = true
	regions.gpuCacheOK = false

	n := Mesh().Size()
	c := Mesh().CellSize()
	dx := (float64(n[2]/2) - 0.5) * c[2]
	dy := (float64(n[1]/2) - 0.5) * c[1]
	dz := (float64(n[0]/2) - 0.5) * c[0]

	for i := 0; i < n[0]; i++ {
		z := float64(i)*c[0] - dz
		for j := 0; j < n[1]; j++ {
			y := float64(j)*c[1] - dy
			for k := 0; k < n[2]; k++ {
				x := float64(k)*c[2] - dx
				if s(x, y, z) { // inside
					regions.arr[i][j][k] = byte(id)
				}
			}
		}
	}
}

func checkRegionIdx(id int) {
	if id < 0 || id > NREGION {
		log.Fatalf("region id should be 0-255, have: %v", id)
	}
}

// Set the region of one cell
func (r *Regions) SetCell(ix, iy, iz int, region int) {
	checkRegionIdx(region)
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

// Get returns the regions as a slice of floats, so it can be output.
func (r *Regions) Get() (*data.Slice, bool) {
	s := data.NewSlice(1, r.Mesh())
	l := s.Host()[0]
	for i := range l {
		l[i] = float32(r.cpu[i])
	}
	return s, false
}

// Re-interpret a contiguous array as a multi-dimensional array of given size.
func resizeBytes(array []byte, size [3]int) [][][]byte {
	Nx, Ny, Nz := size[0], size[1], size[2]
	util.Argument(Nx*Ny*Nz == len(array))
	sliced := make([][][]byte, Nx)
	for i := range sliced {
		sliced[i] = make([][]byte, Ny)
	}
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = array[(i*Ny+j)*Nz+0 : (i*Ny+j)*Nz+Nz]
		}
	}
	return sliced
}

func (r *Regions) Mesh() *data.Mesh { return &globalmesh }
