package cuda

import (
	"math"

	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func GetHopfIndex_SolidAngleFourier(m *data.Slice, mesh *data.Mesh) float64 {

	// Get emergent magnetic field in real space using Berg-Lüscher lattice method
	N := m.Size()
	util.Argument(m.Size() == N)
	F := NewSlice(3, N)
	defer F.Free()
	SetEmergentMagneticField_SolidAngle(F, m, mesh)

	// Rescale field to dimensionless length units by multiplying by cell dimensions
	cellsize := mesh.CellSize()
	cfg := make3DConf(N)
	k_scaleemergentfield_async(F.DevPtr(X), F.DevPtr(Y), F.DevPtr(Z),
		F.DevPtr(X), F.DevPtr(Y), F.DevPtr(Z),
		float32(cellsize[X]), float32(cellsize[Y]), float32(cellsize[Z]),
		N[X], N[Y], N[X], cfg)

	// Initialise FFT plan
	fftPlan := newFFT3DR2C(N[X], N[Y], N[Z])

	// Declare buffers to store FFT
	Nc := fftR2COutputSizeFloats(N)
	fftRBufX := NewSlice(1, N)
	fftCBufX := NewSlice(1, Nc)
	fftRBufY := NewSlice(1, N)
	fftCBufY := NewSlice(1, Nc)
	fftRBufZ := NewSlice(1, N)
	fftCBufZ := NewSlice(1, Nc)
	defer fftRBufX.Free()
	defer fftCBufX.Free()
	defer fftRBufY.Free()
	defer fftCBufY.Free()
	defer fftRBufZ.Free()
	defer fftCBufZ.Free()

	fftRBufX = F.Comp(X)
	fftRBufY = F.Comp(Y)
	fftRBufZ = F.Comp(Z)

	// Perform FFT on each component
	fftPlan.ExecAsync(fftRBufX, fftCBufX)
	fftPlan.ExecAsync(fftRBufY, fftCBufY)
	fftPlan.ExecAsync(fftRBufZ, fftCBufZ)

	// Reconstruct full array using Hermitian symmetry F(-k_x, -k_y, -k_z) = F(k_x, k_y, k_z)^*
	full_array_N := [3]int{2 * N[X], N[Y], N[Z]} // 2 as real + complex part
	Fx_k := NewSlice(1, full_array_N)
	Fy_k := NewSlice(1, full_array_N)
	Fz_k := NewSlice(1, full_array_N)
	k_solidanglefourierfield_async(fftCBufX.DevPtr(0), fftCBufY.DevPtr(0), fftCBufZ.DevPtr(0),
		Fx_k.DevPtr(0), Fy_k.DevPtr(0), Fz_k.DevPtr(0),
		N[X], N[Y], N[Z], cfg)

	summand := NewSlice(1, N)
	k_solidanglefouriersummand_async(summand.DevPtr(0), Fx_k.DevPtr(0), Fy_k.DevPtr(0), Fz_k.DevPtr(0), N[X], N[Y], N[Z], cfg)

	return (1. / (2 * math.Pi * float64(N[X]) * float64(N[Y]) * float64(N[Z]))) * float64(Sum(summand))

}
