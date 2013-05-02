package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"fmt"
	"github.com/barnex/cuda5/cu"
	"html/template"
	"net/http"
	"os"
)

var (
	ui      = &guistate{Steps: 1000, Runtime: 1e-9}
	uitempl = template.Must(template.New("gui").Parse(templText))
)

// http handler that serves main gui
func gui(w http.ResponseWriter, r *http.Request) {
	injectAndWait(func() { util.FatalErr(uitempl.Execute(w, ui)) })
}

type guistate struct {
	Steps               int
	Runtime             float64
	running, pleaseStop bool // todo: mv out of struct
}

func (s *guistate) Time() float32                 { return float32(Time) }
func (s *guistate) ImWidth() int                  { return ui.Mesh().Size()[2] }
func (s *guistate) ImHeight() int                 { return ui.Mesh().Size()[1] }
func (s *guistate) Mesh() *data.Mesh              { return &global_mesh }
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
func (s *guistate) MemInfo() string { f, t := cu.MemGetInfo(); return fmt.Sprint(f/mib, "/", t/mib) }

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
	<title>mx3</title>
	<style media="all" type="text/css">
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

<script>
	function httpGet(url){
    	var xmlHttp = new XMLHttpRequest();
    	xmlHttp.open("GET", url, false);
    	xmlHttp.send(null);
    	return xmlHttp.responseText;
    }
	var running = false
	function updateRunning(){
		try{
			running = (httpGet("/running/") === "true")
		}catch(err){
			running = false	
		}
		if(running){
			document.getElementById("running").innerHTML = "<font color=green><b>Running</b></font>"
		}else{
			document.getElementById("running").innerHTML = "<font color=red><b>Not running</b></font>"
		}
		document.getElementById("breakbutton").disabled = !running
	}
	setInterval(updateRunning, 200)
</script>

<div> <h2> solver </h2>

<table><tr><td>  

	<form action=/ctl/run method="POST">
        <input id=text size=8 name="value" value="{{.Runtime}}"> s <input type="submit" value="Run"/>
	</form>
	<form  action=/ctl/steps method="POST">
        <input id=text size=8 name="value" value="{{.Steps}}"> <input type="submit" value="Steps"/>
	</form>

	<form id=text action=/ctl/pause method="POST"> 
		<input id="breakbutton" type="submit" value="Break"/>
	</form>

	<br/>

</td><td>  
 &nbsp; &nbsp; &nbsp;
</td><td>  

	<span id="running"><font color=red><b>Not running</b></font></span> 
	<span id="dash"> </span>

</td></tr></table>



<script>
	function updateDash(){
		try{
			document.getElementById("dash").innerHTML = httpGet("/dash/")
		}catch(err){}
	}
	function updateDashIfRunning(){
		if(running){
			updateDash();
		}
	}
	updateDash();
	setInterval(updateDashIfRunning, 200);
</script>

<hr/> </div>


<script>
function hide(id) {
    document.getElementById(id).style.display = 'none';
}
function show(id) {
    document.getElementById(id).style.display = 'block';
}
</script>


<h2> display </h2>
<div id=div_display> 

<script>
	var renderQuant = "m"; // TODO: don't forget on reload.
	var renderComp = ""; 
	var img = new Image();
	img.src = "/render/" + renderQuant;

	function updateImg(){
		img.src = "/render/" + renderQuant + "?" + new Date(); // date = cache breaker
		document.getElementById("display").src = img.src;
	}

	function updateImgAsync(){
		if(running && img.complete){
			updateImg();
		}
	}

	setInterval(updateImgAsync, 500);

	function renderSelect() {
		var list=document.getElementById("renderList");
		renderQuant=list.options[list.selectedIndex].text;
		list=document.getElementById("renderComp");
		renderComp=list.options[list.selectedIndex].text;
		updateImg();
	}
</script>

<form>
Display: <select id="renderList" onchange="renderSelect()">
		{{range $k, $v := .Quants}}
  			<option>{{$k}}</option>
		{{end}}
	</select>
	<select id="renderComp" onchange="renderSelect()">
  		<option></option>
  		<option>x</option>
  		<option>y</option>
  		<option>z</option>
	</select>
</form>
<script>
	document.getElementById("renderList").value = renderQuant;
</script>



<img id="display" src="/render/m" width={{.ImWidth}} height={{.ImHeight}} alt="display"/>

<form  action=/setm/ method="POST">
	<b>Re-initialize from .dump file:</b> <input id="text" size=60 name="value" value="{{.Pwd}}"> <input type="submit" value="Submit"/> 
</form>


</div><hr/>


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
	<input type="submit" value="Submit"/>
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
		<td> {{index .WorldNm 0 | printf "%.2f" }} </td> <td> x  </td>
		<td> {{index .WorldNm 1 | printf "%.2f" }} </td> <td> x  </td>
		<td> {{index .WorldNm 2 | printf "%.2f" }} </td> <td> nm3 </td>
	</tr>
</table>
	<input type="submit" value=" Submit"/> <b> Changing the mesh requires some re-initialization time</b>
</form>

<hr/></div>




<div> <h2> process </h2>
	<form action=/ctl/kill  method="POST"> 
		<table>
		<tr><td> <font color=red><b> Kill process:</b></font> </td><td> <input type="submit" value="Kill"/> </td></tr>
		<tr><td> <b>GPU</b> </td><td> {{.Device.Name}} </td></tr>
		</table>
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
