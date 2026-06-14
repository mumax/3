package engine

import (
	"math"
	"strings"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/util"
)

func init() {
	DeclFunc("ext_GeomEdge", ext_GeomEdge, "Returns an edge (+x, -x, +y, -y, +z, -z) of the geometry as a Shape. The Shape may not update automatically after changes to the mesh or geometry; rerun the function to refresh.")
}

func ext_GeomEdge(axis string) Shape {

	s, r := geometry.Slice()
	if r {
		defer cuda.Recycle(s)
	}

	arr3d := s.HostCopy().Scalars()

	n := Mesh().Size()
	Nx, Ny, Nz := n[X], n[Y], n[Z]
	PBC := Mesh().PBC()

	edgemask := make([]bool, Nx*Ny*Nz)

	var offx, offy, offz int
	switch strings.ToLower(axis) {
	case "+x":
		offx, offy, offz = 1, 0, 0
	case "-x":
		offx, offy, offz = -1, 0, 0
	case "+y":
		offx, offy, offz = 0, 1, 0
	case "-y":
		offx, offy, offz = 0, -1, 0
	case "+z":
		offx, offy, offz = 0, 0, 1
	case "-z":
		offx, offy, offz = 0, 0, -1
	default:
		util.Log("GeometryEdge: invalid direction '" + axis + "'. Use +x, -x, +y, -y, +z, or -z")
	}

	for k := 0; k < Nz; k++ {
		for j := 0; j < Ny; j++ {
			for i := 0; i < Nx; i++ {

				if arr3d[k][j][i] == 0 {
					continue
				}

				nx, ny, nz := wrapPBC(i+offx, X), wrapPBC(j+offy, Y), wrapPBC(k+offz, Z)
				if nx < 0 || nx >= Nx || ny < 0 || ny >= Ny || nz < 0 || nz >= Nz || arr3d[nz][ny][nx] == 0 {
					edgemask[(k*Ny+j)*Nx+i] = true
				}
			}
		}
	}

	if PBC[X] != 0 || PBC[Y] != 0 || PBC[Z] != 0 { // Account for PBC in case of Grid resize by repeating GeomEdge
		d := Mesh().CellSize()
		Rx := float64(Nx) * d[X] * float64(PBC[X]&1)
		Ry := float64(Ny) * d[Y] * float64(PBC[Y]&1)
		Rz := float64(Nz) * d[Z] * float64(PBC[Z]&1)
		return maskToShape(edgemask).Repeat(Rx, Ry, Rz)
	} else { // No PBC
		return maskToShape(edgemask)
	}
}

// Helper function to wrap bool mask into a Shape
func maskToShape(mask []bool) Shape {
	n := Mesh().Size()
	Nx, Ny, Nz := n[X], n[Y], n[Z]
	d := Mesh().CellSize()

	return func(x, y, z float64) bool {
		Lx := float64(Nx) * d[X]
		Ly := float64(Ny) * d[Y]
		Lz := float64(Nz) * d[Z]

		ix := int(math.Floor((x + 0.5*Lx) / d[X]))
		iy := int(math.Floor((y + 0.5*Ly) / d[Y]))
		iz := int(math.Floor((z + 0.5*Lz) / d[Z]))

		if ix < 0 || ix >= Nx || iy < 0 || iy >= Ny || iz < 0 || iz >= Nz {
			return false
		}
		return mask[(iz*Ny+iy)*Nx+ix]
	}
}
