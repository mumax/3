package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

// TODO: have newBlaBla() add the quantity to world when it starts with uppercase

var (
	Aex    ScalarParam // inter-cell exchange stiffness in J/m
	lex2   symmparam   // inter-cell exchange length squared * 1e18
	B_exch adderQuant  // exchange field (T) output handle
	E_exch = newGetScalar("E_exch", "J", GetExchangeEnergy)
)

func init() {
	Aex = scalarParam("Aex", "J/m", func(r int) {
		lex2.SetInterRegion(r, r, safediv(2e18*Aex.GetRegion(r), Msat.GetRegion(r)))
	})
	world.LValue("Aex", &Aex)
	world.Func("setLexchange", SetLExchange)
	world.ROnly("B_exch", &B_exch)
	world.ROnly("E_exch", &E_exch)
	world.Func("sign", sign)
}

func initExchange() {
	B_exch = adder(3, Mesh(), "B_exch", "T", func(dst *data.Slice) {
		cuda.AddExchange(dst, M.buffer, lex2.Gpu(), regions.Gpu())
	})
	Quants["B_exch"] = &B_exch
	registerEnergy(GetExchangeEnergy)
}

// Returns the current exchange energy in Joules.
// Note: the energy is defined up to an arbitrary constant,
// ground state energy is not necessarily zero or comparable
// to other simulation programs.
func GetExchangeEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_exch) / Mu0
	// note: M_full is in Tesla, hence /Mu0
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
