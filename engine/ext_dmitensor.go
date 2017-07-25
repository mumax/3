package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	"unsafe"
)

var (
	B_dmi     = NewVectorField("B_dmi", "T", "DMI field", AddDMIField)
	E_dmi     = NewScalarValue("E_dmi", "J", "Total dmi energy", GetDMIEnergy)
	Edens_dmi = NewScalarField("Edens_dmi", "J/m3", "Total dmi energy density", AddDMIEnergyDensity)
	DMItensor = new(dmitensor)
)

var AddDMIEnergyDensity = makeEdensAdder(&B_dmi, -0.5)

func init() {
	registerEnergy(GetDMIEnergy, AddDMIEnergyDensity)
	DeclFunc("ext_DMItensor", SetDMItensor, "Sets DMItensor element (i,j,k)")
}

type dmitensor struct {
	lut    [27]float32
	gpu    cuda.LUTPtr
	gpu_ok bool
}

func (d *dmitensor) Gpu() cuda.LUTPtr {
	if !d.gpu_ok {
		d.upload()
	}
	return d.gpu
}

// D_ijk m_j \partial_i m_k
func (d *dmitensor) SetElement(i, j, k int, value float32) {
	if i > 3 || i < 0 || j > 3 || j < 0 || k > 3 || k < 0 {
		panic(UserErr("invalid DMI tenser index"))
	}

	I := 9*i + 3*j + k
	d.lut[I] = value
	d.gpu_ok = false
}

func (d *dmitensor) upload() {
	// alloc if  needed
	if d.gpu == nil {
		d.gpu = cuda.LUTPtr(cuda.MemAlloc(int64(len(d.lut)) * cu.SIZEOF_FLOAT32))
	}
	lut := d.lut // Copy, to work around Go 1.6 cgo pointer limitations.
	cuda.MemCpyHtoD(unsafe.Pointer(d.gpu), unsafe.Pointer(&lut[0]), cu.SIZEOF_FLOAT32*int64(len(d.lut)))
	d.gpu_ok = true
}

func (d *dmitensor) isZero() bool {
	for i := range d.lut {
		if d.lut[i] != 0 {
			return false
		}
	}
	return true
}

func SetDMItensor(i, j, k int, value float64) {
	DMItensor.SetElement(i, j, k, float32(value))
}

// Adds the current DMI field to dst
func AddDMIField(dst *data.Slice) {
	if !DMItensor.isZero() {
		cuda.AddDMItensor(dst, M.Buffer(), DMItensor.Gpu(), Msat.MSlice(), regions.Gpu(), M.Mesh())
	}
}

// Returns the current DMI energy in Joules.
func GetDMIEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_dmi)
}
