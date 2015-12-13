package engine

var (
	B_ext        excitation
	Edens_zeeman = NewScalarField("Edens_Zeeman", "J/m3", AddEdens_zeeman)
	E_Zeeman     = NewScalarValue("E_Zeeman", "J", "Zeeman energy", GetZeemanEnergy)
)

var AddEdens_zeeman = makeEdensAdder(&B_ext, -1)

func init() {
	Export(Edens_zeeman, "Zeeman energy density")
	B_ext.init("B_ext", "T", "Externally applied field")
	registerEnergy(GetZeemanEnergy, AddEdens_zeeman)
}

func GetZeemanEnergy() float64 {
	return -1 * cellVolume() * dot(&M_full, &B_ext)
}
