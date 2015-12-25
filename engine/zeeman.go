package engine

var (
	B_ext        = NewExcitation("B_ext", "T", "Externally applied field")
	Edens_zeeman = NewScalarField("Edens_Zeeman", "J/m3", "Zeeman energy density", AddEdens_zeeman)
	E_Zeeman     = NewScalarValue("E_Zeeman", "J", "Zeeman energy", GetZeemanEnergy)
)

var AddEdens_zeeman = makeEdensAdder(B_ext, -1)

func init() {
	registerEnergy(GetZeemanEnergy, AddEdens_zeeman)
}

func GetZeemanEnergy() float64 {
	return -1 * cellVolume() * dot(&M_full, B_ext)
}
