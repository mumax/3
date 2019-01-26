package engine

const templText = `
<!DOCTYPE html>
<html>

<head>

	<meta http-equiv="Content-Type" content="text/html; charset=utf-8">

	<title> mumax3 </title>

` + CSS + `

	<script> ; </script>

	{{.JS}}

	<script>
		// Toggle visibility of element with id
		function toggle(id) {
	       var el = document.getElementById(id);
	       if(el.style.display != 'none'){
	          el.style.display = 'none';
			} else {
	          el.style.display = 'block';
			}
	    }
	</script>



	<style type="text/css">
		* {box-sizing: border-box;}
		.img-loupe-div { position: relative; }
		.img-loupe {
		  position: absolute;
		  border: 1px solid #000;
		  border-radius: 2% 50% 50%;
		  box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.4) ;
		  cursor: crosshair;
		  width: 150px;
		  height: 150px;
		  display: none;
		}
	</style>
	<script type="text/javascript">
		// image Loupe
		function updateLoupe(imgID, zoom) {
		var img, loupe;
		img = document.getElementById(imgID);
		loupe = img.parentElement.getElementsByClassName("img-loupe")[0];
		loupe.style.backgroundSize = (img.width * zoom) + "px " + (img.height * zoom) + "px";
		}

		function magnify(imgID, zoom) {
		  var img, loupe, w, h;
		  img = document.getElementById(imgID);

		  /* Create loupe: */
		  loupe = document.createElement("DIV");
		  loupe.setAttribute("class", "img-loupe");
		  img.parentElement.insertBefore(loupe, img);
		  img.style.cursor="crosshair";

		  /* Set background for loupe: */
		  loupe.style.backgroundImage = "url('" + img.src + "')";
		  loupe.style.backgroundRepeat = "no-repeat";
		  loupe.style.backgroundSize = (img.width * zoom) + "px " + (img.height * zoom) + "px";

		  /* Execute a function when someone moves the magnifier loupe over the image: */
		  loupe.addEventListener("mousemove", moveMagnifier);
		  img.addEventListener("mousemove", moveMagnifier);

		  function moveMagnifier(e) {
		    var pos, a, x = 0, y = 0;
		    /* Prevent any other actions that may occur when moving over the image */
		    e.preventDefault();
		    /* Calculate the cursor coordinates relative to the image: */
		    e = e || window.event;
		    a = img.getBoundingClientRect();
		    x = e.pageX - a.left - window.pageXOffset;
		    y = e.pageY - a.top - window.pageYOffset;

		    /* Prevent the magnifier loupe from being positioned outside the image: */
		    loupe.style.display='block';
		    if (x > img.width || x < 0 || y > img.height || y < 0) {loupe.style.display='none';}

		    /* Set the position of the magnifier loupe: */
		    loupe.style.left = (x-10) + "px";
		    loupe.style.top = (y+5) + "px";

		    /* Display what the magnifier loupe "sees": */
		    w = loupe.offsetWidth / 2;
			h = loupe.offsetHeight / 2;
		    loupe.style.backgroundPosition = "-" + ((x * zoom) - w) + "px -" + ((y * zoom) - h) + "px";
		  }
		}
	</script>

</head>


<body>
<iframe src="http://mumax.github.io/header.html" width="100%" height="180" frameborder="0"></iframe>
<span style="color:gray; font-weight:bold; font-size:1.5em" >
	{{.Span "title" "mumax3"}} &nbsp; </span>
	{{.Progress "progress" 100 0}} {{.Span "busy" "" }} &nbsp; {{.ErrorBox}} <br/>
	<hr/>


<script>
	// auto scroll the console window down unless focused.
	var console_focus = false;
	function console_scrolldown(){
		if (!console_focus){
			var textarea = document.getElementById('console');
			textarea.scrollTop = textarea.scrollHeight;
		}
	}

	function setConsoleText(text){
		var textarea = document.getElementById('console'); // id ignored
		textarea.innerHTML = text;
		console_scrolldown();
	}

	// CLI history
	var clihistory = new Array();
	var histindex = 0;

	function clikeydown(event){
		var cli = document.getElementById('cli');
		var key = event.keyCode;
		if (key == 13 && cli.value != ""){ // return key
			notifyel('cli', 'value');
			var backup = cli.value;
			cli.value = "";
			// history code goes last: prevent flackyness of messing up what came before
			clihistory.push(backup);
			histindex = clihistory.length;
		}
		if (key == 38){ // up key
			if (histindex > 0) { histindex--; }
			if (clihistory[histindex] != undefined) { cli.value = clihistory[histindex]; }
		}
		if (key == 40){ // down key
			if (histindex < clihistory.length-1) { histindex++; }
			if (clihistory[histindex] != undefined) { cli.value = clihistory[histindex]; }
		}
	}	
</script>



{{.Data.Div "console"}}

{{.Console "console" 16 84 "" "onfocus=\"console_focus=true\"" "onblur=\"console_focus=false\"" "onmouseover=\"console_focus=true\"" "onmouseout=\"console_focus=false\"" "readonly" "style=\"font-family:monospace; font-size:0.8em;\"" }}	<br/>

{{.CliBox "cli" "" "onkeydown=\"clikeydown(event);\"" "placeholder=\"type commands here, or up/down\"" "size=86" "style=\"font-family:monospace; font-size:0.8em;\""  }}

</div>


{{.Data.Div "mesh"}}

		<table>
			<tr> <td>gridsize: </td> <td>{{.TextBox "nx" "" "size=8"}} </td> <td> &times; {{.TextBox "ny" "" "size=8"}}</td> <td> &times; {{.TextBox "nz" "" "size=8"}}</td> <td>  cells            </td> </tr>
			<tr> <td>cellsize: </td> <td>{{.TextBox "cx" "" "size=8"}} </td> <td> &times; {{.TextBox "cy" "" "size=8"}}</td> <td> &times; {{.TextBox "cz" "" "size=8"}}</td> <td>  m<sup>3</sup>    </td> </tr>
			<tr> <td>PBC:      </td> <td>{{.TextBox "px" "" "size=8"}} </td> <td> &times; {{.TextBox "py" "" "size=8"}}</td> <td> &times; {{.TextBox "pz" "" "size=8"}}</td> <td>  repetitions </td> </tr>
			<tr> <td>worldsize:</td> <td>{{.Span    "wx" ""}} </td> <td> &times; {{.Span    "wy" ""}}</td> <td> &times; {{.Span    "wz" ""}}</td> <td>  nm<sup>3</sup> </td> </tr>
		</table>

		{{.Button "setmesh" "update"}} {{.Span "setmeshwarn" ""}}

</div>



{{.Data.Div "geometry"}}

SetGeom( {{.Data.Shapes | .SelectArray "geomselect" "Universe"}} {{.TextBox "geomargs" "()" }} ) {{.Button "setgeom" "Set"}} </br>
{{.Span "geomdoc" "" "style=\"color:gray\""}}

</div>


{{.Data.Div "initial m"}}

m = {{.Data.Configs | .SelectArray "mselect" "Uniform"}} {{.TextBox "margs" "(1, 0, 0)" }} {{.Button "setm" "Set"}} </br>
{{.Span "mdoc" "" "style=\"color:gray\""}}

</div>


{{.Data.Div "solver"}}

	Type: {{.Select "solvertype" "rk45" "bw_euler" "euler" "heun" "rk4" "rk23" "rk45" "rkf56"}}
	<table>
		<tr> <td>

	<table>
		<tr title="{{.Data.Doc "Run"   }}"> <td> {{.Button "run"   "Run"  }}</td> <td> {{.TextBox "runtime"   1e-9  "size=8"}}s</td></tr> 
		<tr title="{{.Data.Doc "Steps" }}"> <td> {{.Button "steps" "Steps"}}</td> <td> {{.TextBox "runsteps" "1000" "size=8"}} </td></tr>
		<tr title="{{.Data.Doc "Relax" }}"> <td> {{.Button "relax" "Relax"}}</td> <td></td>                                         </tr>
		<tr title="Break current run loop"> <td> {{.Button "break" "Break"}}</td> <td></td>                                         </tr>
	</table>

	</td><td>
		&nbsp; &nbsp; &nbsp; &nbsp;
	</td><td>

	<table>
		<tr title="Time steps taken">   <td>step:    </td><td>{{.Span "nsteps"  "0"}}   </td></tr>
        <tr title="{{.Data.Doc "t"}}">  <td>time:    </td><td>{{.Span "time"    "0"}} s </td></tr>
		<tr title="{{.Data.Doc "dt"}}"> <td>dt:      </td><td>{{.Span "dt"      "0"}} s </td></tr>
		<tr title="Maximum relative error/step"> <td>err/step: </td><td>{{.Span "lasterr" "0"}}   </td></tr>
		<tr title="Maximum absolute torque">     <td>MaxTorque:</td><td>{{.Span "maxtorque" "--"}} </td></tr> 
	</table>

	</td><td>
		&nbsp; &nbsp; &nbsp; &nbsp;
	</td><td>

	<table>
		<tr title="{{.Data.Doc "FixDt" }}"> <td>fixdt:   </td><td>{{.TextBox "fixdt"  "0" "size=8"}} s   </td></tr>
		<tr title="{{.Data.Doc "MinDt" }}"> <td>mindt:   </td><td>{{.TextBox "mindt"  "0" "size=8"}} s   </td></tr>
		<tr title="{{.Data.Doc "MaxDt" }}"> <td>maxdt:   </td><td>{{.TextBox "maxdt"  "0" "size=8"}} s   </td></tr>
		<tr title="{{.Data.Doc "MaxErr"}}"> <td>maxerr:  </td><td>{{.TextBox "maxerr" "0" "size=8"}}/step</td></tr>
	</table>

		</td></tr>
	</table>
</div>





{{.Data.Div "display"}}

<p> 
	Quantity: {{.Data.QuantNames | .SelectArray "renderQuant" "m"}} {{.Select "renderComp" "" "" "x" "y" "z"}} {{.Span "renderDoc" "" "style=\"color:gray\""}} <br/>
	<table>
		<tr title="Slice through z layers">               <td> Slice: {{.Range "renderLayer" 0 0 0 }}  </td><td> {{.Span "renderLayerLabel" "0"}}    </td></tr>
		<tr title="Zoom out large images">                <td> Scale: {{.Range "renderScale" 0 31 31}} </td><td> {{.Span "renderScaleLabel" "1/1"}}  </td></tr>
	</table>
</p>

<p> 
<div class="img-loupe-div">
{{.Img "display" "/render/m" "alt=\"display\""}}
<script>
	/* Set loupe and update it as the image can change... */
	magnify("display", 5);
	setInterval(function () { updateLoupe("display", 5); }, tick*2);
</script>
</div>
</p>


</div>



{{.Data.Div "gnuplot"}}
<p title="{{$.Data.Doc "TableAutoSave"}}">
	TableAutosave: {{.TextBox "tableAutoSave" "0" }}  s 
</p>
		Plot of "table.txt", provided table is being autosaved and gnuplot installed.<br>
		<b>plot "table.txt" using {{.TextBox "usingx" "1"}} : {{.TextBox "usingy" "2"}} with lines </b><br/>
		<p class=ErrorBox>{{.Span "plotErr" ""}}</p>
		{{.Img "plot" "/plot/"}}

</div>




{{.Data.Div "parameters"}}

Region: {{.Number "region" -1 255 -1}} </br>

<table>
{{range .Data.Parameters}}
<tr title="{{$.Data.Doc .}}"> <td>{{.}}</td> <td> {{$.TextBox . ""}} {{$.Data.UnitOf . }}</td> </tr>
{{end}}
</table>

</div>



<hr/>

<div style="font-size:0.9em; color:gray; text-align:center">

{{.Data.Version}} <br/>
{{.Data.GPUInfo}} ({{.Span "memfree" ""}} MB free) <br/>
&copy; 2013 Arne Vansteenkiste, DyNaMat LAB, UGent.


</div>


</body>
</html>
`

const CSS = `
	<style media="all" type="text/css">

		body  { margin-left: 5%; margin-right:5%; font-family: sans-serif; font-size: 14px; }
		img   { margin: 10px; }
		table { border-collapse: collapse; }
		td        { padding: 1px 5px; }
		hr        { border-style: none; border-top: 1px solid #CCCCCC; }
		a         { color: #375EAB; text-decoration: none; }
		div       { margin-left: 20px; margin-top: 5px; margin-bottom: 20px; }
		div#footer{ color:gray; font-size:14px; border:none; }
		.ErrorBox { color: red; font-weight: bold; font-size: 1em; } 
		.TextBox  { border:solid; border-color:#BBBBBB; border-width:1px; padding-left:4px; }
		textarea  { border:solid; border-color:#BBBBBB; border-width:1px; padding-left:4px; color:gray; font-size: 1em; }
	</style>
`
