package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

var (
	Ku1     ScalarParam // Uniaxial anisotropy strength (J/m³)
	ku1_red ScalarParam // Ku1 / Msat (T), auto updated from Ku1 (TODO: form msat)
	AnisU   VectorParam // Uniaxial anisotropy axis
	B_uni   adderQuant  // field due to uniaxial anisotropy output handle
)

func initAnisotropy() {
	AnisU = vectorParam("anisU", "")
	Ku1 = scalarParam("Ku1", "J/m3")
	ku1_red = scalarParam("ku1_red", "T")
	Ku1.post_update = func(region int) {
		ku1_red.setRegion(region, Ku1.GetRegion(region)/Msat.GetRegion(region))
	}

	B_uni = adder(3, Mesh(), "B_uni", "T", func(dst *data.Slice) {
		//TODO: conditionally
		cuda.AddUniaxialAnisotropy(dst, M.buffer, ku1_red.Gpu(), AnisU.Gpu(), regions.Gpu())
	})
	Quants["B_uni"] = &B_uni
	Quants["Ku1"] = &Ku1
	Quants["anisU"] = &AnisU
}
