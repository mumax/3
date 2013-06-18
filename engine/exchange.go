package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

func init() {
	world.LValue("Aex", &Aex)
	B_exch_addr := &B_exch
	world.ROnly("B_exch", &B_exch_addr)
	world.LValue("ExchangeMask", &ExchangeMask)
	Aex = scalarParam("Aex", "J/m", func(r int) {
		lex2.SetInterRegion(r, r, safediv(2e18*Aex.GetRegion(r), Msat.GetRegion(r)))
	})
	world.Func("setLexchange", SetLExchange)
}

var (
	Aex          ScalarParam        // inter-cell exchange stiffness in J/m
	lex2         symmparam          // inter-cell exchange length squared * 1e18
	ExchangeMask staggeredMaskQuant // Mask that scales Aex/Msat between cells.
	B_exch       adderQuant         // exchange field (T) output handle
)

func initExchange() {
	B_exch = adder(3, Mesh(), "B_exch", "T", func(dst *data.Slice) {
		cuda.AddExchange(dst, M.buffer, lex2.Gpu(), regions.Gpu())
	})
	Quants["B_exch"] = &B_exch

	ExchangeMask = staggeredMask(Mesh(), "exchangemask", "")
	Quants["exchangemask"] = &ExchangeMask
}

func SetLExchange(region1, region2 int, exchangelength float64) {
	l2 := (exchangelength * exchangelength) * 1e18
	lex2.SetInterRegion(region1, region2, l2)
}
