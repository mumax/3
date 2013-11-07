package engine

var (
	B_ext    excitation
	E_Zeeman *GetScalar
)

func init() {
	B_ext.init("B_ext", "T", "Externally applied field")
	E_Zeeman = NewGetScalar("E_Zeeman", "J", "Zeeman energy", getZeemanEnergy)
	registerEnergy(getZeemanEnergy)
}

func getZeemanEnergy() float64 {
	return -1 * cellVolume() * dot(&M_full, &B_ext)
}
