package engine

// THIS FILE IS AUTO GENERATED FROM webgui.js
// EDITING IS FUTILE

const templText = `
<!DOCTYPE html>
<html>
<head>
	<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
	<title>mx3</title>
	<style media="all" type="text/css">
		body { margin: 40px; font-family: Helvetica, Arial, sans-serif; font-size: 15px; }
		img  { margin: 10px; }
		h1   { font-size: 28px; color: gray; }
		h2   { font-size: 20px; color: gray; }
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
	function http(method, url){
    	var xmlHttp = new XMLHttpRequest();
    	xmlHttp.open(method, url, false);
    	xmlHttp.send(null);
    	return xmlHttp.responseText;
    }
	function httpGet(url){
		return http("GET", url);
	}
	function httpPost(url){
		return http("POST", url);
	}

	var running = false;
	function updateRunning(){
		try{
			running = (httpGet("/running/") === "true");
		}catch(err){
			running = false	;
		}
		if(running){
			document.getElementById("running").innerHTML = "<font color=green><b>Running</b></font>";
		}else{
			document.getElementById("running").innerHTML = "<font color=red><b>Not running</b></font>";
		}
	}
	setInterval(updateRunning, 200);

	function rpc(command){
		httpPost("/script/" + command);
	}

	function rpcBox(label, command, args){
		var par = document.scripts[document.scripts.length - 1].parentNode;

		var boxes = [];
		for (var i=0; i < args.length; i++){
			var textbox = document.createElement("input");
			textbox.type = "text";
			textbox.value = args[i];
			textbox.id = "text";
			textbox.size = 10;
			par.appendChild(textbox);
			boxes.push(textbox);
		}

		var button = document.createElement("input");
		button.type = "button";
		button.value = label;
		button.onclick = function(){
			var args = "(";
			for (var i=0; i < boxes.length; i++){
				args += boxes[i].value;
				if (i != boxes.length - 1) { args += ","; }
			}
			args += ")";
			rpc(command + args);
		};
		par.appendChild(button);
		par.appendChild(document.createElement("br"));
	}

	// add a button that executes an rpc command
	function rpcButton(label, command){
		rpcBox(label, command, [])
	}

</script>


<div> <h2> solver </h2>

<table><tr><td>  

	<script> rpcBox("Run", "run", [1e-9]);     </script>
	<script> rpcBox("Steps", "steps", [1000]); </script>
	<script> rpcButton("Break", "pause");    </script>

</td><td>  
 &nbsp; &nbsp; &nbsp;
</td><td>  

	<span id="running"><font color=red><b>Not running</b></font></span> 
	<span id="dash"> </span>

</td></tr></table>



<script>
	function updateDash(){
		try{
			document.getElementById("dash").innerHTML = httpGet("/dash/");
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
		var renderQuantComp = renderQuant;
		if(renderComp != ""){
			renderQuantComp += "/" + renderComp;
		}	
		img.src = "/render/" + renderQuantComp + "?" + new Date(); // date = cache breaker
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


<div> <h2> <font color=red> Danger zone </font> </h2>

	<form  action=/setm/ method="POST">
		<b>Upload m from file:</b> <input id="text" size=60 name="value" value="{{.Pwd}}"> <input type="submit" value="Submit"/> 
	</form>

	<form action=/ctl/kill method="POST"> 
		<b> Kill process:</b> <input type="submit" value="Kill"/> 
	</form>

<hr/></div>

<div id="footer">
<center>
{{.Uname}}<br/>
Copyright 2012-2013 <a href="mailto:a.vansteenkiste@gmail.com">Arne Vansteenkiste</a>, <a href="http://dynamat.ugent.be">Dynamat group</a>, Ghent University, Belgium.<br/>
You are free to modify and distribute this software under the terms of the <a href="http://www.gnu.org/licenses/gpl-3.0.html">GPLv3 license</a>.
</center>
</div>

</body>
</html>
`
