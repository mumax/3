package engine

var (
	B_ext    excitation
	E_Zeeman = NewGetScalar("E_Zeeman", "J", getZeemanEnergy)
)

func init() {
	World.LValue("B_ext", &B_ext, "External field (T)")
	World.ROnly("E_Zeeman", &E_Zeeman, "Zeeman energy (J)")
}

func initBExt() {
	B_ext.init(Mesh(), "B_ext", "T")
	registerEnergy(getZeemanEnergy)
}

func getZeemanEnergy() float64 {
	return -1 * cellVolume() * dot(&M_full, &B_ext)
}
