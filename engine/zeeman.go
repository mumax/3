package engine

var (
	B_ext    excitation
	E_Zeeman = newGetScalar("E_Zeeman", "J", GetZeemanEnergy)
)

func init() {
	world.LValue("B_ext", &B_ext)
	world.ROnly("E_Zeeman", &E_Zeeman)
}

func initBExt() {
	B_ext.init(Mesh(), "B_ext", "T")
	registerEnergy(GetZeemanEnergy)
}

func GetZeemanEnergy() float64 {
	return -1 * cellVolume() * dot(&M_full, &B_ext) / Mu0
}
