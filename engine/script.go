package engine

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/script"
	"code.google.com/p/mx3/util"
	"log"
	"os"
)

var parser = script.NewParser()

// Runs a script file.
func RunFile(fname string) {
	// first we compile the entire file into an executable tree
	f, err := os.Open(fname)
	util.FatalErr(err)
	defer f.Close()
	code, err2 := parser.Parse(f)
	util.FatalErr(err2)

	// now the parser is not used anymore so it can handle web requests
	goServe(*flag_port)

	// start executing the tree, possibly injecting commands from web gui
	for _, cmd := range code {
		cmd.Eval()
	}
}

// Compile file but do not run it. Used to check for errors.
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
	parser.AddFunc("steps", Steps)
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

func doAutosave(what interface {
	Autosave(float64)
}, period float64) {
	what.Autosave(period)
}

func doSaveTable(period float64) {
	Table.Autosave(period)
}
