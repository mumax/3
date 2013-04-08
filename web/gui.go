package web

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/engine"
	"code.google.com/p/mx3/util"
	"html/template"
	"io/ioutil"
	"net/http"
)

var (
	guiTempl *template.Template
	guis     = new(guistate)
)

func gui(w http.ResponseWriter, r *http.Request) {
	// TODO: racy with engine init, mv to package engine
	if guiTempl == nil {
		guiTempl = loadTemplate("gui.html") // TODO: embed.
		guis.Heun = engine.Solver
		guis.Mesh = engine.GetMesh()
	}
	util.FatalErr(guiTempl.Execute(w, guis))
}

type guistate struct {
	*cuda.Heun
	*data.Mesh
}

func (s *guistate) Time() float32 { return float32(engine.Time) }

func loadTemplate(fname string) *template.Template {
	body, err := ioutil.ReadFile(fname)
	util.FatalErr(err)
	return template.Must(template.New(fname).Parse(string(body)))
}
