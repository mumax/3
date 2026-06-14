package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	Ext_HopfIndex_TwoPointStencil             = NewScalarValue("ext_hopfindex_twopointstencil", "", "Hopf index calculated using two-point stencil", GetHopfIndex_TwoPointStencil)
	Ext_HopfIndexDensity_TwoPointStencil      = NewScalarField("ext_hopfindexdensity_twopointstencil", "1/m3", "Hopf index density calculated using two-point stencil", SetHopfIndexDensity_TwoPointStencil)
	Ext_EmergentMagneticField_TwoPointStencil = NewVectorField("ext_emergentmagneticfield_twopointstencil", "1/m2", "Emergent magnetic field calculated using two-point stencil", SetEmergentMagneticField_TwoPointStencil)

	Ext_HopfIndex_FivePointStencil             = NewScalarValue("ext_hopfindex_fivepointstencil", "", "Hopf index calculated using five-point stencil", GetHopfIndex_FivePointStencil)
	Ext_HopfIndexDensity_FivePointStencil      = NewScalarField("ext_hopfindexdensity_fivepointstencil", "1/m3", "Hopf index density calculated using five-point stencil", SetHopfIndexDensity_FivePointStencil)
	Ext_EmergentMagneticField_FivePointStencil = NewVectorField("ext_emergentmagneticfield_fivepointstencil", "1/m2", "Emergent magnetic field calculated using five-point stencil", SetEmergentMagneticField_FivePointStencil)

	Ext_HopfIndex_SolidAngle             = NewScalarValue("ext_hopfindex_solidangle", "", "Hopf index calculated using Berg-Lüscher lattice method", GetHopfIndex_SolidAngle)
	Ext_HopfIndexDensity_SolidAngle      = NewScalarField("ext_hopfindexdensity_solidangle", "1/m3", "Hopf index density computed using Berg-Lüscher lattice method", SetHopfIndexDensity_SolidAngle)
	Ext_EmergentMagneticField_SolidAngle = NewVectorField("ext_emergentmagneticfield_solidangle", "1/m2", "Emergent magnetic field computed using Berg-Lüscher lattice method", SetEmergentMagneticField_SolidAngle)

	Ext_HopfIndex_SolidAngleFourier = NewScalarValue("ext_hopfindex_solidanglefourier", "", "Hopf index calculated using Berg-Lüscher lattice method to calculate emergent field, with emergent field Fourier transformed", GetHopfIndex_SolidAngleFourier)
)

func GetHopfIndex_TwoPointStencil() float64 {
	Refer("Knapman2025")
	h := ValueOf(Ext_HopfIndexDensity_TwoPointStencil)
	defer cuda.Recycle(h)
	c := Mesh().CellSize()
	return -c[X] * c[Y] * c[Z] * float64(cuda.Sum(h))
}

func SetHopfIndexDensity_TwoPointStencil(dst *data.Slice) {
	Refer("Knapman2025")
	cuda.SetHopfIndexDensity_TwoPointStencil(dst, M.Buffer(), M.Mesh())
}

func SetEmergentMagneticField_TwoPointStencil(dst *data.Slice) {
	cuda.SetEmergentMagneticField_TwoPointStencil(dst, M.Buffer(), M.Mesh())
}

func GetHopfIndex_FivePointStencil() float64 {
	Refer("Knapman2025")
	h := ValueOf(Ext_HopfIndexDensity_FivePointStencil)
	defer cuda.Recycle(h)
	c := Mesh().CellSize()
	return -c[X] * c[Y] * c[Z] * float64(cuda.Sum(h))
}

func SetHopfIndexDensity_FivePointStencil(dst *data.Slice) {
	Refer("Knapman2025")
	cuda.SetHopfIndexDensity_FivePointStencil(dst, M.Buffer(), M.Mesh())
}

func SetEmergentMagneticField_FivePointStencil(dst *data.Slice) {
	cuda.SetEmergentMagneticField_FivePointStencil(dst, M.Buffer(), M.Mesh())
}

func GetHopfIndex_SolidAngle() float64 {
	Refer("Knapman2025")
	h := ValueOf(Ext_HopfIndexDensity_SolidAngle)
	defer cuda.Recycle(h)
	c := Mesh().CellSize()
	return -c[X] * c[Y] * c[Z] * float64(cuda.Sum(h))
}

func SetHopfIndexDensity_SolidAngle(dst *data.Slice) {
	Refer("Knapman2025")
	cuda.SetHopfIndexDensity_SolidAngle(dst, M.Buffer(), M.Mesh())
}

func SetEmergentMagneticField_SolidAngle(dst *data.Slice) {
	cuda.SetEmergentMagneticField_SolidAngle(dst, M.Buffer(), M.Mesh())
}

func GetHopfIndex_SolidAngleFourier() float64 {
	Refer("Knapman2025")
	return cuda.GetHopfIndex_SolidAngleFourier(M.Buffer(), M.Mesh())
}
