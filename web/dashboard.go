package web

//import (
//	"code.google.com/p/mx3/engine"
//	"code.google.com/p/mx3/util"
//	"html/template"
//	"net/http"
//)
//
//// handles "/dash", that shows solver state
//func dashHandler(w http.ResponseWriter, r *http.Request) {
//	engine.InjectAndWait(func() { util.FatalErr(dashtempl.Execute(w, ui)) })
//}
//
//var dashtempl = template.Must(template.New("dash").Parse(dashText))
//
//const dashText = `
//<table>
//<tr><td> step:        </td><td> {{.Solver.NSteps}} </td><td> &nbsp; &nbsp; evaluations:</td><td> {{.Solver.NEval}}</td></tr>
//<tr><td> time:        </td><td> {{printf "%12e" .Time}}         s</td><td> &nbsp; &nbsp; time step:   </td><td> {{printf "%12e" .Solver.Dt_si}} s</td></tr>
//<tr><td> max err/step:</td><td> {{printf "%e" .Solver.MaxErr}} </td><td> &nbsp; &nbsp; err/step:    </td><td> {{printf "%12e" .Solver.LastErr}}</td></tr>
//</table>
//`
