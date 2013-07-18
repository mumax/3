package engine

// THIS FILE IS AUTO GENERATED FROM gui.html
// EDITING IS FUTILE

const templText = `
<!DOCTYPE html>
<html>

<head>

	<meta http-equiv="Content-Type" content="text/html; charset=utf-8">

	<title>mx3 GUI</title>

	<style media="all" type="text/css">

		body  { margin: 20px; font-family: Ubuntu, Arial, sans-serif; font-size: 14px; color: #444444}
		h1    { font-size: 22px; font-weight: normal; color: black}
		h2    { font-size: 18px; font-weight: normal; color: black}
		img   { margin: 15px; }
		table { border:"10"; }
		hr    { border-style: none; border-top: 1px solid #CCCCCC; }

		a     { color: #375EAB; text-decoration: none; }
		a#hide{ color: black; font-size:17px; text-decoration: none; cursor: pointer; font-weight: normal; }

		div       { margin-left: 20px; margin-top: 10px; margin-bottom: 20px; }
		div#header{ color:gray; font-size:16px; }
		div#footer{ color:gray; font-size:14px; }

		.ErrorBox { color: red; font-weight: bold; } 
		.TextBox  { border:solid; border-color:#BBBBBB; border-width:1px; padding-left:4px;}
	</style>

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

	{{.JS}}

</head>


<body>

		<h1> mx 3 GUI </h1> 
	<div id=header>
		<p> {{.ErrorBox}} </p>
	</div><hr/>

	
	<h2> geometry </h2><div>

		<table>
			<tr> <td>gridsize: </td> <td>{{.Span "nx" "0"}} </td> <td>x {{.Span "ny" "0"}}</td> <td>x {{.Span "nz" "0"}}</td> </tr>
			<tr> <td>cellsize: </td> <td>{{.Span "cx" "0"}} </td> <td>x {{.Span "cy" "0"}}</td> <td>x {{.Span "cz" "0"}}</td> </tr>
			<tr> <td>worldsize:</td> <td>{{.Span "wx" "0"}} </td> <td>x {{.Span "wy" "0"}}</td> <td>x {{.Span "wz" "0"}}</td> </tr>
		</table>
	</div><hr/>


	<h2> control </h2><div>
	</div><hr/>



	<div id="footer">
		<br/>
		<center>
			mumax 3<br/>
			Copyright 2012-2013 <a href="mailto:a.vansteenkiste@gmail.com">Arne Vansteenkiste</a>,
			<a href="http://dynamat.ugent.be">Dynamat group</a>, Ghent University, Belgium.<br/>
			You are free to modify and distribute this software under the terms of the
			<a href="http://www.gnu.org/licenses/gpl-3.0.html">GPLv3 license</a>.
		</center>
	</div>

</body>
</html>
`
