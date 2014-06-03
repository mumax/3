package engine

var (
	B_ext        excitation
	E_Zeeman     *GetScalar
	Edens_zeeman sAdder
)

func init() {
	B_ext.init("B_ext", "T", "Externally applied field")
	E_Zeeman = NewGetScalar("E_Zeeman", "J", "Zeeman energy", GetZeemanEnergy)
	Edens_zeeman.init("Edens_Zeeman", "J/m3", "Zeeman energy density", makeEdensAdder(&B_ext, -1))
	registerEnergy(GetZeemanEnergy, Edens_zeeman.AddTo)
}

func GetZeemanEnergy() float64 {
	return -1 * cellVolume() * dot(&M_full, &B_ext)
}
