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
		h1    { font-size: 22px; color: gray; }
		h2    { font-size: 18px; color: gray; }
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

</head>


<body>

	<span style="color:gray; font-weight:bold; font-size:1.5em"> {{.Data.Title}} &nbsp; &nbsp; </span> {{.UpdateButton ""}} {{.UpdateBox "live"}} &nbsp; &nbsp; {{.ErrorBox}} <br/>
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
</script>


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


{{.Data.Div "console"}}

{{.TextArea "console" 8 84 "" "onfocus=\"console_focus=true\"" "onblur=\"console_focus=false\"" "onmouseover=\"console_focus=true\"" "onmouseout=\"console_focus=false\"" "readonly" "style=\"font-family:monospace; font-size:0.8em;\"" }}	<br/>

{{.TextBox "cli" "" "placeholder=\"type commands here\"" "size=86" "style=\"font-family:monospace; font-size:0.8em;\""}}

</div>


{{.Data.Div "geometry"}}

</div>

</body>
</html>
`
