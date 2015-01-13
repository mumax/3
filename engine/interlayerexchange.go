package engine

// Interlayer exchange coupling (IEC)
import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// Anisotropy variables
var (
	J_linear, J_quadratic         ScalarParam  // bilinear and biquadratic coupling constants
	toplayer, bottomlayer	ScalarParam // top- and bottom layer positions
	top_red, bottom_red 	derivedParam
	J1_red, J2_red              derivedParam // J1 / Msat and J2/Msat
	exchange_direction	    VectorParam // Direction of interlayer exchange
	B_iec                vAdder     // field due to IEC (T)
	E_iec                *GetScalar // IEC energy
	Edens_iec            sAdder     // IEC density
	One 		int
)

func init() {
	J_linear.init("J_linear", "J/m2", "bilinear coupling constant", []derived{&J1_red})
	J_quadratic.init("J_quadratic", "J/m2", "biquadratic coupling constant", []derived{&J2_red})
	toplayer.init("toplayer", "#", "Position of the toplayer",[]derived{&top_red})
	bottomlayer.init("bottomlayer", "#", "Position of the bottomlayer",[]derived{&bottom_red})
	exchange_direction.init("IEC_direction", "", "Direction of Interlayer exchange coupling")
	B_iec.init("B_iec", "T", "interlayer exchange field", AddInterlayerField)
	E_iec = NewGetScalar("E_iec", "J", "interlayer exchange energy", GetInterlayerEnergy)
	Edens_iec.init("Edens_iec", "J/m3", "interlayer exchange energy density", makeEdensAdder(&B_iec, -0.5))
	registerEnergy(GetInterlayerEnergy, Edens_iec.AddTo)

	//J1_red = J1 / Msat
	J1_red.init(1, []updater{&J_linear, &Msat}, func(p *derivedParam) {
		paramDiv(p.cpu_buf, J_linear.cpuLUT(), Msat.cpuLUT())
	})

	//J2_red = J2 / Msat
	J2_red.init(1, []updater{&J_quadratic, &Msat}, func(p *derivedParam) {
		paramDiv(p.cpu_buf, J_quadratic.cpuLUT(), Msat.cpuLUT())
	})
	top_red.init(1, []updater{&toplayer, &Msat}, func(p *derivedParam) {
		return
	})
	bottom_red.init(1, []updater{&bottomlayer, &Msat}, func(p *derivedParam) {
		return
	})

}

// Add the interlayer exchange field to dst
func AddInterlayerField(dst *data.Slice) {
	if !(J1_red.isZero()) || !(J2_red.isZero()) {
		cuda.AddInterlayerExchange(dst, M.Buffer(), J1_red.gpuLUT1(), J2_red.gpuLUT1(), toplayer.gpuLUT1(), bottomlayer.gpuLUT1(), exchange_direction.gpuLUT(), regions.Gpu(), Mesh())
	}
}
/*func AddInterlayerEnergyDensity(dst *data.Slice) {
	buf := cuda.Buffer(B_iec.NComp(), B_iec.Mesh().Size())
	defer cuda.Recycle(buf)

	// unnormalized magnetization:
	Mf, r := M_full.Slice()
	if r {
		defer cuda.Recycle(Mf)
	}
	if !(J1_red.isZero()) || !(J2_red.isZero()) {
		cuda.Zero(buf)	
		cuda.AddInterlayerExchange(buf, M.Buffer(), J1_red.gpuLUT1(), J2_red.gpuLUT1(), toplayer.gpuLUT1(), bottomlayer.gpuLUT1(), regions.Gpu(), Mesh())
		cuda.AddDotProduct(dst, -1./2., buf, Mf)
	}
}

func GetInterlayerEnergy() float64 {
	buf := cuda.Buffer(1, Edens_iec.Mesh().Size())
	defer cuda.Recycle(buf)

	cuda.Zero(buf)
	AddInterlayerEnergyDensity(buf)
	return cellVolume() * float64(cuda.Sum(buf))
}
*/
func GetInterlayerEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_iec)
}

