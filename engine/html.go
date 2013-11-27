package engine

// THIS FILE IS AUTO GENERATED FROM gui.html
// EDITING IS FUTILE

const templText = `
<!DOCTYPE html>
<html>

<head>

	<meta http-equiv="Content-Type" content="text/html; charset=utf-8">

	<title>{{.Data.Title}}</title>

	<style media="all" type="text/css">

		body  { margin-left: 5%; margin-right:5%; font-family: sans-serif; font-size: 14px; }
		img   { margin: 10px; }
		table { border-collapse: collapse; }
		tr:nth-child(even) { background-color: white; }
		tr:nth-child(odd)  { background-color: #F7F7FF; }
		td        { padding: 1px 5px; }
		hr        { border-style: none; border-top: 1px solid #CCCCCC; }
		a         { color: #375EAB; text-decoration: none; }
		div       { margin-left: 20px; margin-top: 5px; margin-bottom: 20px; }
		div#footer{ color:gray; font-size:14px; border:none; }
		.ErrorBox { color: red; font-weight: bold; font-size: 1.5em; } 
		.TextBox  { border:solid; border-color:#BBBBBB; border-width:1px; padding-left:4px; }
		textarea  { border:solid; border-color:#BBBBBB; border-width:1px; padding-left:4px; color:gray; font-size: 1em; }
	</style>

	{{.JS}}

	<script>
		function toggle(id) {
	       var el = document.getElementById(id);
	       if(el.style.display != 'none'){
	          el.style.display = 'none';
			} else {
	          el.style.display = 'block';
			}
	    }
	</script>

</head>


<body>

	<span style="color:gray; font-weight:bold; font-size:1.5em"> {{.Data.Title}} &nbsp; &nbsp; </span>  {{.ErrorBox}} <br/>
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
	setInterval(console_scrolldown, tick);

//	// CLI history
//	var history = new Array();
//	var histindex = 0;
//
//	function clikeydown(){
//		var cli = document.getElementById('cli');
//		var key = event.keyCode;
//		if (key == 13 && cli.value != ""){ // return key
//			history.push(cli.value);
//			histindex = history.length;
//			cli.onchange();
//		}
//		if (key == 38){ // up key
//			if (histindex > 0) { histindex--; }
//			if (history[histindex] != undefined) { cli.value = history[histindex]; }
//		}
//		if (key == 40){ // down key
//			if (histindex < history.length-1) { histindex++; }
//			if (history[histindex] != undefined) { cli.value = history[histindex]; }
//		}
//	}	
</script>



{{.Data.Div "console"}}

{{.TextArea "console" 8 84 "" "onfocus=\"console_focus=true\"" "onblur=\"console_focus=false\"" "onmouseover=\"console_focus=true\"" "onmouseout=\"console_focus=false\"" "readonly" "style=\"font-family:monospace; font-size:0.8em;\"" }}	<br/>

{{.TextBox "cli" "" "placeholder=\"type commands here\"" "size=86" "style=\"font-family:monospace; font-size:0.8em;\""}}

</div>


{{.Data.Div "geometry"}}

		<table>
			<tr> <td>gridsize: </td> <td>{{.TextBox "nx" "" "size=8"}} </td> <td> &times; {{.TextBox "ny" "" "size=8"}}</td> <td> &times; {{.TextBox "nz" "" "size=8"}}</td> <td>  cells             </td> </tr>
			<tr> <td>cellsize: </td> <td>{{.TextBox "cx" "" "size=8"}} </td> <td> &times; {{.TextBox "cy" "" "size=8"}}</td> <td> &times; {{.TextBox "cz" "" "size=8"}}</td> <td>  nm<sup>3</sup>    </td> </tr>
			<tr> <td>PBC:      </td> <td>{{.TextBox "px" "" "size=8"}} </td> <td> &times; {{.TextBox "py" "" "size=8"}}</td> <td> &times; {{.TextBox "pz" "" "size=8"}}</td> <td>  repetitions </td> </tr>
			<tr> <td>worldsize:</td> <td>{{.Span    "wx" ""}} </td> <td> &times; {{.Span    "wy" ""}}</td> <td> &times; {{.Span    "wz" ""}}</td> <td>  nm<sup>3</sup> </td> </tr>
		</table>

		{{.Button "setmesh" "&#x26a0; update"}}

</div>

{{.Data.Div "solver"}}

{{.Span "steps" ""}}

</div>

{{.Data.Div "display"}}

<p> 
{{.Data.QuantNames | .SelectArray "renderQuant" "m"}} {{.Select "renderComp" "" "" "x" "y" "z"}} {{.Span "renderDoc" "m"}} <br/>
    z-slice: {{.Range "renderLayer" 0 0 0 }} zoom out: {{.Range "renderScale" 0 31 31}}
</p>

<p> 
{{.Img "display" "/render/m" "alt=\"display\""}}
</p>


</div>


<hr/>

<div style="font-size:0.9em; color:gray; text-align:center">

{{.Data.Version}} <br/>
&copy; 2013 Arne Vansteenkiste, DyNaMat LAB, UGent.


</div>


</body>
</html>
`
