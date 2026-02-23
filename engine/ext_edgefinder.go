package engine

import "github.com/mumax/3/cuda"

func init() {
	DeclFunc("GeometryEdgePlusX", GeometryEdgePlusX, "Returns the +X edge of the geometry as a Shape")
	DeclFunc("GeometryEdgeMinusX", GeometryEdgeMinusX, "Returns the -X edge of the geometry as a Shape")
	DeclFunc("GeometryEdgePlusY", GeometryEdgePlusY, "Returns the +Y edge of the geometry as a Shape")
	DeclFunc("GeometryEdgeMinusY", GeometryEdgeMinusY, "Returns the -Y edge of the geometry as a Shape")
	DeclFunc("GeometryEdgePlusZ", GeometryEdgePlusZ, "Returns the +Z edge of the geometry as a Shape")
	DeclFunc("GeometryEdgeMinusZ", GeometryEdgeMinusZ, "Returns the -Z edge of the geometry as a Shape")
}

// Each of these calls GeometryEdges once and returns the corresponding Shape

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
	// Get GPU slice
	slice, recycle := geometry.Slice()
	if recycle {
		defer cuda.Recycle(slice)
	}

	// Single CPU copy
	host := slice.HostCopy()
	arr3d := host.Scalars() // [Nz][Ny][Nx]

	n := geometry.Mesh().Size()
	Nx, Ny, Nz := n[X], n[Y], n[Z]

	// Preallocate all 6 edge masks
	edgeMasks := make([][]bool, 6)
	for i := range edgeMasks {
		edgeMasks[i] = make([]bool, Nx*Ny*Nz)
	}

	// Directions as dx, dy, dz
	directions := [6][3]int{
		{1, 0, 0},  // +X
		{-1, 0, 0}, // -X
		{0, 1, 0},  // +Y
		{0, -1, 0}, // -Y
		{0, 0, 1},  // +Z
		{0, 0, -1}, // -Z
	}

	// Compute all edge masks
	for iz := 0; iz < Nz; iz++ {
		for iy := 0; iy < Ny; iy++ {
			for ix := 0; ix < Nx; ix++ {
				if arr3d[iz][iy][ix] == 0 {
					continue // skip empty
				}

				for dirIdx, dir := range directions {
					nx, ny, nz := ix+dir[0], iy+dir[1], iz+dir[2]
					if nx < 0 || nx >= Nx || ny < 0 || ny >= Ny || nz < 0 || nz >= Nz || arr3d[nz][ny][nx] == 0 {
						edgeMasks[dirIdx][(iz*Ny+iy)*Nx+ix] = true
					}
				}
			}
		}
	}

	return maskToShape(edgeMasks[0], Nx, Ny, Nz), maskToShape(edgeMasks[1], Nx, Ny, Nz),
		maskToShape(edgeMasks[2], Nx, Ny, Nz), maskToShape(edgeMasks[3], Nx, Ny, Nz),
		maskToShape(edgeMasks[4], Nx, Ny, Nz), maskToShape(edgeMasks[5], Nx, Ny, Nz)
}

// Helper to wrap mask into a Shape
func maskToShape(mask []bool, Nx, Ny, Nz int) Shape {
	return func(x, y, z float64) bool {
		d := geometry.Mesh().CellSize()
		Lx := float64(Nx) * d[X]
		Ly := float64(Ny) * d[Y]
		Lz := float64(Nz) * d[Z]

		ix := int((x + 0.5*Lx) / d[X])
		iy := int((y + 0.5*Ly) / d[Y])
		iz := int((z + 0.5*Lz) / d[Z])

		if ix < 0 || ix >= Nx || iy < 0 || iy >= Ny || iz < 0 || iz >= Nz {
			return false
		}
		return mask[(iz*Ny+iy)*Nx+ix]
	}
}
