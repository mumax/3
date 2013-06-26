package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"log"
)

func init() {
	world.Func("SetGeom", SetGeometry)
	world.Func("DefRegion", DefRegion)
	world.ROnly("regions", &regions)
}

var (
	regions Regions       // global regions map
	geom    Shape   = nil // nil means universe
)

const MAXREG = 256 // maximum number of regions

type Regions struct {
	arr        [][][]byte   // regions map: cell i,j,k -> byte index
	cpu        []byte       // arr data, stored contiguously
	gpuCache   *cuda.Bytes  // gpu copy of cpu data, possibly out-of-sync
	gpuCacheOK bool         // gpuCache in sync with cpu
	defined    [MAXREG]bool // has region i been defined already (not allowed to set it if not defined)
	autosave
}

func (r *Regions) init() {
	mesh := Mesh() // global sim mesh

	r.autosave.nComp = 1
	r.autosave.name = "regions"
	r.autosave.mesh = mesh
	r.cpu = make([]byte, mesh.NCell())
	r.arr = resizeBytes(r.cpu, mesh.Size())
	r.gpuCache = cuda.NewBytes(mesh)

	DefRegion(1, universe)
}

// Define a region with id (0-255) to be inside the Shape.
func DefRegion(id int, s Shape) {
	if id < 0 || id > MAXREG {
		log.Fatalf("region id should be 0-255, have: %v", id)
	}
	regions.defined[id] = true
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
	M.stencilGeom() // TODO: revise if really needed
	regions.gpuCacheOK = false
	regions.defined[id] = true
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

func SetGeometry(s Shape) {
	geom = s
	regions.rasterGeom()
}

// Rasterises the global geom shape
// TODO: deduplicate from DefRegion
func (r *Regions) rasterGeom() {
	s := geom
	if s == nil {
		s = universe
	}

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
					if regions.arr[i][j][k] == 0 {
						regions.arr[i][j][k] = 1
					}
				} else {
					regions.arr[i][j][k] = 0
				}
			}
		}
	}
	M.stencilGeom() // TODO: revise if really needed
	regions.gpuCacheOK = false
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
