package engine

// Interlayer exchange coupling (IEC)
import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// Anisotropy variables
var (
	J_linear    = NewScalarParam("J_linear", "J/m2", "Bilinear coupling constant", &J1_red)
	J_quadratic = NewScalarParam("J_quadratic", "J/m2", "Biquadratic coupling constant", &J2_red)

	toplayer           = NewScalarParam("toplayer", "#", "Position of the toplayer")
	bottomlayer        = NewScalarParam("bottomlayer", "#", "Position of the bottomlayer")
	exchange_direction = NewVectorParam("IEC_direction", "", "Direction of Interlayer exchange coupling")

	J1_red, J2_red DerivedParam // J1/Msat and J2/Msat

	B_iec     = NewVectorField("B_iec", "T", "Interlayer exchange field", AddInterlayerField)
	Edens_iec = NewScalarField("Edens_iec", "J/m3", "Interlayer exchange energy density", AddInterlayerEnergyDensity)
	E_iec     = NewScalarValue("E_iec", "J", "Interlayer exchange energy", GetInterlayerEnergy)
)

var AddInterlayerEnergyDensity = makeEdensAdder(&B_iec, -0.5)

func init() {

	J_linear.addChild(&J1_red)
	J_quadratic.addChild(&J2_red)

	registerEnergy(GetInterlayerEnergy, AddInterlayerEnergyDensity)

	//J1_red = J1 / Msat
	J1_red.init(SCALAR, []parent{J_linear, Msat}, func(p *DerivedParam) {
		paramDiv(p.cpu_buf, J_linear.cpuLUT(), Msat.cpuLUT())
	})

	//J2_red = J2 / Msat
	J2_red.init(SCALAR, []parent{J_quadratic, Msat}, func(p *DerivedParam) {
		paramDiv(p.cpu_buf, J_quadratic.cpuLUT(), Msat.cpuLUT())
	})

}

// Add the interlayer exchange field to dst
func AddInterlayerField(dst *data.Slice) {
	if J1_red.nonZero() && J2_red.nonZero() {
		cuda.AddInterlayerExchange(dst, M.Buffer(), J1_red.gpuLUT1(), J2_red.gpuLUT1(), toplayer.gpuLUT1(), bottomlayer.gpuLUT1(), exchange_direction.gpuLUT(), regions.Gpu(), Mesh())
	}
}

func GetInterlayerEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_iec)
}
