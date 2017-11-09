package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	Ground		= NewScalarParam("Ground", "a.u.", "Ground")
	Terminal	= NewScalarParam("Terminal", "a.u.", "Terminal")
	volt		voltage   // Local voltage
	Jmask     vectorfield // mask for J from V
)

func init() {
	DeclROnly("Voltage", AsScalarField(&volt), "Voltage (V)")
	DeclFunc("SolveVoltage", SolveVoltage, "Calculates V,J")
	DeclROnly("Jmask", &Jmask, "Mask for J")
}

func SolveVoltage(iterations int) {
	volt.update()
	Jmask.update()
	ground := Ground.MSlice()
	defer ground.Recycle()
	terminal := Terminal.MSlice()
	defer terminal.Recycle()

	for iter:=0;iter<iterations;iter++{
		cuda.CalculateVolt(M.Buffer(), volt.voltage, ground ,terminal,M.Mesh())
	}
	cuda.CalculateMaskJ(Jmask.data,M.Buffer(), volt.voltage,M.Mesh())
}

type voltage struct {
	voltage     *data.Slice      // data buffer
}

func (b *voltage) update() {
	if b.voltage == nil {
		b.voltage = cuda.NewSlice(b.NComp(), b.Mesh().Size())
	}
}

func (b *voltage) Mesh() *data.Mesh       { return Mesh() }
func (b *voltage) NComp() int             { return 1 }
func (b *voltage) Name() string           { return "Voltage" }
func (b *voltage) Unit() string           { return "V" }
func (b *voltage) average() []float64     { return qAverageUniverse(b) }
func (b *voltage) EvalTo(dst *data.Slice) { EvalTo(b, dst) }
func (b *voltage) Slice() (*data.Slice, bool) {
	b.update()
	return b.voltage, false
}


type vectorfield struct {
	data     *data.Slice      // data buffer
}

func (b *vectorfield) update() {
	if b.data == nil {
		b.data = cuda.NewSlice(b.NComp(), b.Mesh().Size())
	}
}

func (b *vectorfield) Mesh() *data.Mesh       { return Mesh() }
func (b *vectorfield) NComp() int             { return 3 }
func (b *vectorfield) Name() string           { return "Mask J" }
func (b *vectorfield) Unit() string           { return "A.U." }
func (b *vectorfield) average() []float64     { return qAverageUniverse(b) }
func (b *vectorfield) EvalTo(dst *data.Slice) { EvalTo(b, dst) }
func (b *vectorfield) Slice() (*data.Slice, bool) {
	b.update()
	return b.data, false
}