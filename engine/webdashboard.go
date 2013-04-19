package engine

import (
	"code.google.com/p/mx3/util"
	"html/template"
	"net/http"
)

var dashtempl = template.Must(template.New("dash").Parse(dashText))

func dash(w http.ResponseWriter, r *http.Request) {
	injectAndWait(func() { util.FatalErr(dashtempl.Execute(w, ui)) })
}

const dashText = `
<table> 
<tr><td> step:        </td><td> {{.Solver.NSteps}} </td><td> &nbsp; &nbsp; undone steps:</td><td> {{.Solver.NUndone}}</td></tr>  
<tr><td> time:        </td><td> {{.Time}}         s</td><td> &nbsp; &nbsp; time step:   </td><td> {{.Solver.Dt_si}} s</td></tr>  
<tr><td> max err/step:</td><td> {{.Solver.MaxErr}} </td><td> &nbsp; &nbsp; err/step:    </td><td> {{.Solver.LastErr}}</td></tr>  
</table>
`
