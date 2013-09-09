package engine

var (
	B_ext    excitation
	E_Zeeman = NewGetScalar("E_Zeeman", "J", "Zeeman energy", getZeemanEnergy)
)

func init() {
	B_ext.init("B_ext", "T", "Externally applied field")
	registerEnergy(getZeemanEnergy)
}

func getZeemanEnergy() float64 {
	return -1 * cellVolume() * dot(&M_full, &B_ext)
}
