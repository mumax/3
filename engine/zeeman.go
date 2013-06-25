package engine

var (
	B_ext excitation
)

func init() {
	world.LValue("B_ext", &B_ext)
}

func initBExt() {
	B_ext.init(Mesh(), "B_ext", "T")
	registerEnergy(ZeemanEnergy)
}

func ZeemanEnergy() float64 {
	return -1 * Volume() * dot(&M_full, &B_ext) / Mu0
}
