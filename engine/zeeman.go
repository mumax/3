package engine

var (
	B_ext excitation
	//E_Zeeman = NewGetScalar("E_Zeeman", "J", getZeemanEnergy)
)

func init() {
	DeclLValue("B_ext", &B_ext, "External field (T)")
	//DeclROnly("E_Zeeman", &E_Zeeman, "Zeeman energy (J)")
	B_ext.init(&globalmesh, "B_ext", "T")
	registerEnergy(getZeemanEnergy)
}

func getZeemanEnergy() float64 {
	return -1 * cellVolume() * dot(&M_full, &B_ext)
}
