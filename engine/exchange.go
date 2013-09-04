package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

var (
	Aex    ScalarParam                       // inter-cell exchange stiffness in J/m
	B_exch adderQuant                        // exchange field (T) output handle
	Dex    = scalarParam("dmi", "J/m2", nil) //func(r int) { // Dzyaloshinskii-Moriya strength in J/m²
	lex2   symmparam                         // inter-cell exchange length squared * 1e18
	//E_exch = NewGetScalar("E_exch", "J", getExchangeEnergy) // Exchange (normal+DM) energy in J
)

func init() {
	Aex = scalarParam("Aex", "J/m", func(r int) {
		lex2.SetInterRegion(r, r, safediv(2e18*Aex.GetRegion(r), Msat.GetRegion(r)))
	})
	DeclLValue("Aex", &Aex, "Exchange stiffness (J/m)")
	DeclFunc("setLexchange", SetLExchange, "Sets inter-material exchange length between two regions.")

	//DeclROnly("E_exch", &E_exch, "Exchange energy (J)")
	DeclFunc("sign", sign, "Signum function")

	DeclLValue("Dex", &Dex, "Dzyaloshinskii-Moriya strength (J/m²)")

	B_exch.init(3, &globalmesh, "B_exch", "T", "Exchange field (T)", func(dst *data.Slice) {
		if Dex.zero {
			cuda.AddExchange(dst, M.buffer, lex2.Gpu(), regions.Gpu())
		} else {
			// DMI only implemented for uniform parameters
			// interaction not clear with space-dependent parameters
			msat := Msat.GetUniform()
			D := Dex.GetUniform() / msat
			A := Aex.GetUniform() / msat
			cuda.AddDMI(dst, M.buffer, float32(D), float32(A)) // dmi+exchange
		}
	})
	registerEnergy(getExchangeEnergy)
}

// Returns the current exchange energy in Joules.
// Note: the energy is defined up to an arbitrary constant,
// ground state energy is not necessarily zero or comparable
// to other simulation programs.
func getExchangeEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_exch)
}

// Defines the exchange coupling between different regions by specifying the
// exchange length of the interaction between them.
// 	lex := sqrt(2*Aex / Msat)
// In case of different materials it is not always clear what the exchange
// between them should be, especially if they have different Msat. By specifying
// the exchange length, it is up to the user to decide which Msat to use.
// When using regions, there is by default no exchange coupling between different regions.
// A negative length may be specified to obtain antiferromagnetic coupling.
func SetLExchange(region1, region2 int, exlen float64) {
	l2 := sign(exlen) * (exlen * exlen) * 1e18
	lex2.SetInterRegion(region1, region2, l2)
}

func sign(x float64) float64 {
	switch {
	case x > 0:
		return 1
	case x < 0:
		return -1
	default:
		return 0
	}
}
