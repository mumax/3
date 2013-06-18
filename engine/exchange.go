package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

func init() {
	world.Var("aex", &Aex)
	B_exch_addr := &B_exch
	world.ROnly("B_exch", &B_exch_addr)
	world.LValue("ExchangeMask", &ExchangeMask)
}

var (
	Aex          symmparam          // inter-cell exchange stiffness in J/m
	ExchangeMask staggeredMaskQuant // Mask that scales Aex/Msat between cells.
	B_exch       adderQuant         // exchange field (T) output handle
)

func initExchange() {
	B_exch = adder(3, Mesh(), "B_exch", "T", func(dst *data.Slice) {
		cuda.AddExchange(dst, M.buffer, Aex.Gpu(), regions.Gpu())
	})
	Quants["B_exch"] = &B_exch

	ExchangeMask = staggeredMask(Mesh(), "exchangemask", "")
	Quants["exchangemask"] = &ExchangeMask
}
