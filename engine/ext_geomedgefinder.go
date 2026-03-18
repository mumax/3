package engine

import (
	"math"

	"github.com/mumax/3/cuda"
)

func init() {
	DeclFunc("GeometryEdgePlusX", GeometryEdgePlusX, "Returns the +X edge of the geometry as a Shape")
	DeclFunc("GeometryEdgeMinusX", GeometryEdgeMinusX, "Returns the -X edge of the geometry as a Shape")
	DeclFunc("GeometryEdgePlusY", GeometryEdgePlusY, "Returns the +Y edge of the geometry as a Shape")
	DeclFunc("GeometryEdgeMinusY", GeometryEdgeMinusY, "Returns the -Y edge of the geometry as a Shape")
	DeclFunc("GeometryEdgePlusZ", GeometryEdgePlusZ, "Returns the +Z edge of the geometry as a Shape")
	DeclFunc("GeometryEdgeMinusZ", GeometryEdgeMinusZ, "Returns the -Z edge of the geometry as a Shape")
}

func GeometryEdgePlusX() Shape {
	plusX, _, _, _, _, _ := GeometryEdges()
	return plusX
}

func GeometryEdgeMinusX() Shape {
	_, minusX, _, _, _, _ := GeometryEdges()
	return minusX
}

func GeometryEdgePlusY() Shape {
	_, _, plusY, _, _, _ := GeometryEdges()
	return plusY
}

func GeometryEdgeMinusY() Shape {
	_, _, _, minusY, _, _ := GeometryEdges()
	return minusY
}

func GeometryEdgePlusZ() Shape {
	_, _, _, _, plusZ, _ := GeometryEdges()
	return plusZ
}

func GeometryEdgeMinusZ() Shape {
	_, _, _, _, _, minusZ := GeometryEdges()
	return minusZ
}

func GeometryEdges() (plusX, minusX, plusY, minusY, plusZ, minusZ Shape) {

	s, r := geometry.Slice()
	if r {
		defer cuda.Recycle(s)
	}

	arr3d := s.HostCopy().Scalars()

	n := Mesh().Size()
	Nx, Ny, Nz := n[X], n[Y], n[Z]

	edgeMasks := make([][]bool, 6)
	for i := range edgeMasks {
		edgeMasks[i] = make([]bool, Nx*Ny*Nz)
	}

	for k := 0; k < Nz; k++ {
		for j := 0; j < Ny; j++ {
			for i := 0; i < Nx; i++ {
				if arr3d[k][j][i] == 0 {
					continue
				}

				neighborX, neighborY, neighborZ := i+1, j, k
				if neighborX < 0 || neighborX >= Nx || arr3d[neighborZ][neighborY][neighborX] == 0 {
					edgeMasks[0][(k*Ny+j)*Nx+i] = true
				}

				neighborX, neighborY, neighborZ = i-1, j, k
				if neighborX < 0 || neighborX >= Nx || arr3d[neighborZ][neighborY][neighborX] == 0 {
					edgeMasks[1][(k*Ny+j)*Nx+i] = true
				}

				neighborX, neighborY, neighborZ = i, j+1, k
				if neighborY < 0 || neighborY >= Ny || arr3d[neighborZ][neighborY][neighborX] == 0 {
					edgeMasks[2][(k*Ny+j)*Nx+i] = true
				}

				neighborX, neighborY, neighborZ = i, j-1, k
				if neighborY < 0 || neighborY >= Ny || arr3d[neighborZ][neighborY][neighborX] == 0 {
					edgeMasks[3][(k*Ny+j)*Nx+i] = true
				}

				neighborX, neighborY, neighborZ = i, j, k+1
				if neighborZ < 0 || neighborZ >= Nz || arr3d[neighborZ][neighborY][neighborX] == 0 {
					edgeMasks[4][(k*Ny+j)*Nx+i] = true
				}

				neighborX, neighborY, neighborZ = i, j, k-1
				if neighborZ < 0 || neighborZ >= Nz || arr3d[neighborZ][neighborY][neighborX] == 0 {
					edgeMasks[5][(k*Ny+j)*Nx+i] = true
				}

			}
		}
	}

	return maskToShape(edgeMasks[0], Nx, Ny, Nz), maskToShape(edgeMasks[1], Nx, Ny, Nz),
		maskToShape(edgeMasks[2], Nx, Ny, Nz), maskToShape(edgeMasks[3], Nx, Ny, Nz),
		maskToShape(edgeMasks[4], Nx, Ny, Nz), maskToShape(edgeMasks[5], Nx, Ny, Nz)
}

// Helper function to wrap bool mask into a Shape
func maskToShape(mask []bool, Nx, Ny, Nz int) Shape {
	return func(x, y, z float64) bool {
		d := Mesh().CellSize()
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
