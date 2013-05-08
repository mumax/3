package engine

import (
	"code.google.com/p/mx3/script"
	"code.google.com/p/mx3/util"
	"fmt"
	"io"
	"log"
)

func RunScript(src io.Reader) {
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

// wrappers to make them callable from script

func (f *ScalFn) Eval() interface{} {
	return (*f)()
}

func (f *ScalFn) Assign(e script.Expr) {
	(*f) = func() float64 { return e.Eval().(float64) }
}

func (f *VecFn) Eval() interface{} {
	return (*f)()
}

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

func cint(f float64) int {
	i := int(f)
	if float64(i) != f {
		panic(fmt.Errorf("need integer, have: %v", f))
	}
	return i
}
