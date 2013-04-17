package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"html/template"
	"net/http"
	"sync"
)

var (
	ui      = &guistate{Steps: 1000, Runtime: 1e-9, Cond: sync.NewCond(new(sync.Mutex))}
	uitempl = template.Must(template.New("gui").Parse(templText))
)

func gui(w http.ResponseWriter, r *http.Request) {
	ui.Lock()
	defer ui.Unlock()
	util.FatalErr(uitempl.Execute(w, ui))
}

type guistate struct {
	Msg                 string
	Steps               int
	Runtime             float64
	Running, pleaseStop bool
	*sync.Cond
}

func (s *guistate) Time() float32    { return float32(Time) }
func (s *guistate) Lock()            { ui.L.Lock() }
func (s *guistate) Unlock()          { ui.L.Unlock() }
func (s *guistate) ImWidth() int     { return ui.Mesh().Size()[2] }
func (s *guistate) ImHeight() int    { return ui.Mesh().Size()[1] }
func (s *guistate) Mesh() *data.Mesh { return &mesh }
func (s *guistate) Uname() string    { return Uname }
func (s *guistate) Version() string  { return VERSION }
func (s *guistate) Solver() *cuda.Heun {
	if Solver == nil {
		return &zeroSolver
	} else {
		return Solver
	}
}

// surrogate solver if no real one is set, provides zero values for time step etc to template.
var zeroSolver cuda.Heun

const templText = `
<!DOCTYPE html>
<html>
<head>
	<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
	{{if .Running}}<meta http-equiv="refresh" content="1">{{end}}
	<title>mx3</title>
	<style media="screen" type="text/css">
		body { margin: 40px; font-family: Helvetica, Arial, sans-serif; font-size: 15px; }
		img  { margin: 10px; }
		h1   { font-size: 28px; font-color: gray; }
		h2   { font-size: 20px; }
		hr   { border-style: none; border-top: 1px solid gray; }
		a    { color: #375EAB; text-decoration: none; }
		table{ border:"20"; }
		input#text{ border:solid; border-color:grey; border-width:1px; padding-left:4px;}
		div#header{ color:gray; font-size:16px; }
		div#footer{ color:gray; font-size:14px; }
	</style>
</head>

<body>

<div id="header"> <h1> {{.Version}} </h1> <hr/> </div>

<div> <h2> solver </h2>

<table><tr><td>  

	<form id=text action=/ctl/pause method="POST"> 
		Status: {{if .Running}}
			<b> Running </b> <input type="submit" value="Pause"/>
		 {{else}}
 			<b> Paused</b> 
		{{end}}
	</form>

	{{if not .Running}}
	<form action=/ctl/run method="POST">
        <input id=text size=8 name="value" value="{{.Runtime}}"> s <input type="submit" value="Run"/>
	</form>
	<form  action=/ctl/steps/ method="POST">
        <input id=text size=8 name="value" value="{{.Steps}}"> <input type="submit" value="Steps"/>
	</form>
	{{end}}
	<br/>
	<form action=/ctl/kill  method="POST"> <font color=red><b>Danger Zone:</b></font> <input type="submit" value="Kill"/> </form>

</td><td>  
 &nbsp; &nbsp; &nbsp;
</td><td>  


<table>
<tr><td> step:        </td><td> {{.Solver.NSteps}} </td></tr> 
<tr><td> undone steps:</td><td> {{.Solver.NUndone}}</td></tr>  
<tr><td> time:        </td><td> {{.Time}}         s</td></tr>  
<tr><td> time step:   </td><td> {{.Solver.Dt_si}} s</td></tr>  
<tr><td> max err/step:</td><td> {{.Solver.MaxErr}} </td></tr>  
<tr><td> err/step:    </td><td> {{.Solver.LastErr}}</td></tr>  
</table>

</td></tr></table> 

<hr/> </div>

<div> <h2> magnetization </h2> 
<a href="/render/m"> <img width={{.ImWidth}} height={{.ImHeight}} src="/render/m"  alt="m"> </a>
<hr/></div>

<div> <h2> parameters </h2> 
	<form action=/setparam/ method="POST">
	<table>
	{{range $k, $v := .Params}}
		<tr><td> {{$k}}: </td><td> 
		{{range $v.Comp}} 
        	<input id=text size=8 name="{{$k}}{{.}}" value="{{$v.GetComp .}}"> 
		{{end}} {{$v.Unit}} <font color=grey>&nbsp;({{$v.Descr}})</font> </td></tr>
	{{end}}
	</table>
	<input type="submit" value=" SAVE "/>
	</form>
<hr/></div>

<div><h2> mesh </h2> 
<form action=/setmesh/ method="POST"><table> 
	<tr>
		<td> grid size: </td>
		<td> <input id=text size=8 name="gridsizex" value="{{index .Mesh.Size 2}}"> </td> <td> x </td>
		<td> <input id=text size=8 name="gridsizey" value="{{index .Mesh.Size 1}}"> </td> <td> x </td>
		<td> <input id=text size=8 name="gridsizez" value="{{index .Mesh.Size 0}}"> </td> <td>   </td>
	</tr>

	<tr>
		<td> cell size: </td>
		<td> <input id=text size=8 name="cellsizex" value="{{index .Mesh.CellSize 2}}"> </td> <td> x  </td>
		<td> <input id=text size=8 name="cellsizey" value="{{index .Mesh.CellSize 1}}"> </td> <td> x  </td>
		<td> <input id=text size=8 name="cellsizez" value="{{index .Mesh.CellSize 0}}"> </td> <td> m3 </td>
	</tr>

	<tr>
		<td> world size: &nbsp;&nbsp; </td>
		<td> {{index .Mesh.WorldSize 2}} </td> <td> x  </td>
		<td> {{index .Mesh.WorldSize 1}} </td> <td> x  </td>
		<td> {{index .Mesh.WorldSize 0}} </td> <td> m3 </td>
	</tr>
</table>
	<input type="submit" value=" SAVE "/> <b> Changing the mesh requires some re-initialization time</b>
</form>

<hr/></div>

<div id="footer">
<center>
{{.Uname}}
</center>
</div>

</body>
</html>
`
