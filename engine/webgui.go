package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
	"html/template"
	"net/http"
	"os"
)

var (
	ui      = &guistate{}
	uitempl = template.Must(template.New("gui").Parse(templText))
)

// http handler that serves main gui
func gui(w http.ResponseWriter, r *http.Request) {
	injectAndWait(func() { util.FatalErr(uitempl.Execute(w, ui)) })
}

type guistate struct{}

func (s *guistate) Time() float32                 { return float32(Time) }
func (s *guistate) ImWidth() int                  { return ui.Mesh().Size()[2] }
func (s *guistate) ImHeight() int                 { return ui.Mesh().Size()[1] }
func (s *guistate) Mesh() *data.Mesh              { return &globalmesh }
func (s *guistate) Uname() string                 { return uname }
func (s *guistate) Version() string               { return VERSION }
func (s *guistate) Pwd() string                   { pwd, _ := os.Getwd(); return pwd }
func (s *guistate) Device() cu.Device             { return cu.CtxGetDevice() }
func (s *guistate) Quants() map[string]downloader { return quants }

// world size in nm.
func (s *guistate) WorldNm() [3]float64 {
	return [3]float64{WorldSize()[X] * 1e9, WorldSize()[Y] * 1e9, WorldSize()[Z] * 1e9}
}

const mib = 1024 * 2014

// TODO: strangely this reports wrong numbers (x2 too low).
// func (s *guistate) MemInfo() string { f, t := cu.MemGetInfo(); return fmt.Sprint(f/mib, "/", t/mib) }

func (s *guistate) Solver() *cuda.Heun {
	if Solver == nil {
		return &zeroSolver
	} else {
		return Solver
	}
}

// surrogate solver if no real one is set, provides zero values for time step etc to template.
var zeroSolver cuda.Heun
