package engine

import (
	"code.google.com/p/mx3/script"
	"code.google.com/p/mx3/util"
	"fmt"
	"io"
	"log"
	"os"
)

// Runs a script file.
func RunFile(fname string) {
	f, err := os.Open(fname)
	util.FatalErr(err)
	defer f.Close()
	runScript(f)
}

func runScript(src io.Reader) {
	p := script.NewParser()

	p.AddFunc("print", myprint)
	p.AddFunc("setmesh", setmeshfloat)

	p.AddFloat("t", &Time)
	p.AddVar("aex", &Aex)
	p.AddVar("msat", &Msat)
	p.AddVar("alpha", &Alpha)
	p.AddVar("b_ext", &B_ext)

	util.FatalErr(p.Exec(src))
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
		v := e.Eval().([]interface{})
		util.Argument(len(v) == 3)
		return [3]float64{v[0].(float64), v[1].(float64), v[2].(float64)}
	}
}

func myprint(msg ...interface{}) {
	log.Println(msg...)
}

func setmeshfloat(nx, ny, nz, cx, cy, cz float64) {
	SetMesh(cint(nx), cint(ny), cint(nz), cx, cy, cz)
}

// safe conversion from float to integer.
func cint(f float64) int {
	i := int(f)
	if float64(i) != f {
		panic(fmt.Errorf("need integer, have: %v", f))
	}
	return i
}
