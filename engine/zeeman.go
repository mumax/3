package engine

var (
	B_ext        excitation
	E_Zeeman     *GetScalar
	Edens_zeeman = ScalarFunc("Edens_Zeeman", "J/m3", AddEdens_zeeman)
)

var AddEdens_zeeman = makeEdensAdder(&B_ext, -1)

func init() {
	Export(Edens_zeeman, "Zeeman energy density")

	B_ext.init("B_ext", "T", "Externally applied field")
	E_Zeeman = NewGetScalar("E_Zeeman", "J", "Zeeman energy", GetZeemanEnergy)
	registerEnergy(GetZeemanEnergy, AddEdens_zeeman)
}

func GetZeemanEnergy() float64 {
	return -1 * cellVolume() * dot(&M_full, &B_ext)
}
