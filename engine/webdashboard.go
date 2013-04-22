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

//util.Dashf("step: % 8d (%6d) t: % 12es Δt: % 12es ε:% 12e", s.NSteps, s.NUndone, Time, s.Dt_si, s.LastErr)
const dashText = `
<table> 
<tr><td> step:        </td><td> {{.Solver.NSteps}} </td><td> &nbsp; &nbsp; undone steps:</td><td> {{.Solver.NUndone}}</td></tr>  
<tr><td> time:        </td><td> {{printf "%12e" .Time}}         s</td><td> &nbsp; &nbsp; time step:   </td><td> {{printf "%12e" .Solver.Dt_si}} s</td></tr>  
<tr><td> max err/step:</td><td> {{printf "%e" .Solver.MaxErr}} </td><td> &nbsp; &nbsp; err/step:    </td><td> {{printf "%12e" .Solver.LastErr}}</td></tr>  
</table>
`
