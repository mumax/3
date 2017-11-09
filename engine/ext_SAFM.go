package engine

// Exchange interaction (Heisenberg + Dzyaloshinskii-Moriya)
// See also cuda/exchange.cu and cuda/dmi.cu

import (
	"github.com/mumax/3/cuda"
//	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
//	"github.com/mumax/3/util"
//	"unsafe"
)

var (
	//AFMex   = NewScalarParam("AFMex", "J/m2", "Antiferromagnetic Exchange stiffness", &AFMex)
	//AFMex=NewScalarValue("AFMex", "J/m2", "Antiferromagnetic Exchange stiffness", &AFMex)
	AFMex float64
	AFMR1 int
	AFMR2 int
	tsp float64
//        AFMex=NewScalarParam("AFMex", "J/m2", "Antiferromagnetic Exchange stiffness",AFMex)
//	AFMR1=NewScalarParam("AFMR1", "a.u.", "Region1")
//	AFMR2=NewScalarParam("AFMR2", "a.u.", "Region2")
	AFMB_exch     = NewVectorField("AFMB_exch", "T", "AFM Exchange field", AddAFMExchangeField)
	AFME_exch     = NewScalarValue("AFME_exch", "J", "Total AFM exchange energy", GetAFMExchangeEnergy)
	AFMEdens_exch = NewScalarField("AFMEdens_exch", "J/m3", "Total AFM exchange energy density", AddAFMExchangeEnergyDensity)

)

var AddAFMExchangeEnergyDensity = makeEdensAdder(&AFMB_exch, -0.5) // TODO: normal func

func init() {
	registerEnergy(GetAFMExchangeEnergy, AddAFMExchangeEnergyDensity)
        DeclVar("AFMex", &AFMex, "Antiferromagnetic Exchange stiffness")
        DeclVar("AFMR1", &AFMR1, "Region1 AFM")
        DeclVar("AFMR2", &AFMR2, "Region1 AFM")
        DeclVar("tsp", &tsp, "Spacer depth")
}

// Adds the current AFMexchange field to dst
func AddAFMExchangeField(dst *data.Slice) {
	Msat := Msat.MSlice()
	defer Msat.Recycle()
	cuda.AddAFMExchange(dst, M.Buffer(), float32(AFMex),AFMR1,AFMR2,float32(tsp), Msat,regions.Gpu(), M.Mesh())
}

// Returns the current exchange energy in Joules.
func GetAFMExchangeEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &AFMB_exch)
}


