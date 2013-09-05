package engine

var (
	B_ext    excitation
	E_Zeeman = NewGetScalar("E_Zeeman", "J", "Zeeman energy", getZeemanEnergy)
)

func init() {
	DeclLValue("B_ext", &B_ext, "External field (T)")
	B_ext.init(&globalmesh, "B_ext", "T")
	registerEnergy(getZeemanEnergy)
}

func getZeemanEnergy() float64 {
	return -1 * cellVolume() * dot(&M_full, &B_ext)
}
