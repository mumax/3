package engine

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/script"
	"code.google.com/p/mx3/util"
	"fmt"
	"io"
	"log"
	"os"
)

//TODO use go/scanner to handle negative numbers etc.

var parser = script.NewParser()

// Runs a script file.
func RunFile(fname string) {
	f, err := os.Open(fname)
	util.FatalErr(err)
	defer f.Close()
	RunScript(f)
}

// Runs script form input.
func RunScript(src io.Reader) {
	util.FatalErr(parser.Exec(src))
}

func Vet(fname string) {
	f, err := os.Open(fname)
	util.FatalErr(err)
	defer f.Close()
	_, err = parser.Parse(f)
	util.FatalErr(err)
}

func init() {
	parser.AddFunc("print", myprint)

	parser.AddFunc("setgridsize", setGridSize)
	parser.AddFunc("setcellsize", setCellSize)

	parser.AddFunc("run", Run)
	parser.AddFunc("autosave", doAutosave)
	parser.AddFunc("savetable", doSaveTable)

	parser.AddFunc("average", average)

	parser.AddFloat("t", &Time)

	parser.AddVar("aex", &Aex)
	parser.AddVar("msat", &Msat)
	parser.AddVar("alpha", &Alpha)
	parser.AddVar("b_ext", &B_ext)
	parser.AddVar("dmi", &DMI)
	parser.AddVar("ku1", &Ku1)
	parser.AddVar("xi", &Xi)
	parser.AddVar("spinpol", &SpinPol)
	parser.AddVar("j", &J)

	parser.AddVar("m", &M)

	//log.Println("parser initialized")
}

// needed only to make it callable from scripts
func (f *ScalFn) Eval() interface{} {
	return (*f)()
}

// needed only to make it callable from scripts
func (f *ScalFn) Assign(e script.Expr) {
	(*f) = func() float64 { return e.Eval().(float64) }
}

// needed only to make it callable from scripts
func (f *VecFn) Eval() interface{} {
	return (*f)()
}

// needed only to make it callable from scripts
func (f *VecFn) Assign(e script.Expr) {
	(*f) = func() [3]float64 {
		return eval3float64(e)
	}
}

// evaluate e and cast return value to [3]float64 (vector)
func eval3float64(e script.Expr) [3]float64 {
	v := e.Eval().([]interface{})
	util.Argument(len(v) == 3)
	return [3]float64{v[0].(float64), v[1].(float64), v[2].(float64)}

}

// needed only to make it callable from scripts
func (b *buffered) Assign(e script.Expr) {
	b.Set(e.Eval().(*data.Slice))
}

// needed only to make it callable from scripts
func (b *buffered) Eval() interface{} {
	return b
}

func myprint(msg ...interface{}) {
	log.Println(msg...)
}

//func setmeshfloat(nx, ny, nz, cx, cy, cz float64) {
//	SetMesh(cint(nx), cint(ny), cint(nz), cx, cy, cz)
//}

func doAutosave(what interface {
	Autosave(float64)
}, period float64) {
	what.Autosave(period)
}

func doSaveTable(period float64) {
	Table.Autosave(period)
}

// safe conversion from float to integer.
func cint(f float64) int {
	i := int(f)
	if float64(i) != f {
		panic(fmt.Errorf("need integer, have: %v", f))
	}
	return i
}
