package engine

import (
	"code.google.com/p/mx3/script"
	"code.google.com/p/mx3/util"
	"io"
)

func (s *ScalFn) Eval() interface{} {
	return (*s)()
}

func (s *ScalFn) Assign(e script.Expr) {
	(*s) = func() float64 { return e.Eval().(float64) }
}

func RunScript(src io.Reader) {
	p := script.NewParser(src)
	p.AddFloat("t", &Time)
	p.AddVar("aex", &Aex)
	p.AddVar("msat", &Msat)
	p.AddVar("alpha", &Alpha)
	util.FatalErr(p.Exec())

}
