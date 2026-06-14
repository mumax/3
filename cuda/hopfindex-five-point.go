package cuda

import (
	"math"

	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func SetHopfIndexDensity_FivePointStencil(h, m *data.Slice, mesh *data.Mesh) {
	N := m.Size()

	// Create buffers to store emergent field F and vector potential A
	F := Buffer(3, N)
	defer Recycle(F)
	A := Buffer(3, N)
	defer Recycle(A)

	// Get Hopf index density F · A
	SetEmergentMagneticField_FivePointStencil(F, m, mesh)
	SetVectorPotential(A, F, mesh)
	AddDotProduct(h, 1.0, F, A)
}

// Sets the emergent magnetic field F_i = (1/8π) ε_{ijk} m · (∂m/∂x_j × ∂m/∂x_k)
// See hopf-emergentmagneticfieldfivepoint.cu
func SetEmergentMagneticField_FivePointStencil(F, m *data.Slice, mesh *data.Mesh) {
	cellsize := mesh.CellSize()
	N := F.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)
	icycz := float32(1.0 / (cellsize[Y] * cellsize[Z]))
	iczcx := float32(1.0 / (cellsize[Z] * cellsize[X]))
	icxcy := float32(1.0 / (cellsize[X] * cellsize[Y]))
	prefactor := float32(1.0 / (8 * math.Pi))

	k_setemergentmagneticfieldfivepoint_async(F.DevPtr(X), F.DevPtr(Y), F.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z), prefactor,
		icycz, iczcx, icxcy, N[X], N[Y], N[Z], mesh.PBC_code(), cfg)
}

// Sets the vector potential A corresponding to the emergent magnetic field F such that F = ∇ × A
// See hopf-vectorpotential.cu
func SetVectorPotential(A, F *data.Slice, mesh *data.Mesh) {
	cellsize := mesh.CellSize()
	N := A.Size()
	util.Argument(F.Size() == N)
	cfg := make3DConf(N)

	k_setvectorpotential_async(A.DevPtr(X), A.DevPtr(Y), A.DevPtr(Z),
		F.DevPtr(X), F.DevPtr(Y), F.DevPtr(Z),
		float32(cellsize[Y]), N[X], N[Y], N[Z], mesh.PBC_code(), cfg)
}
