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
		textarea  { border:solid; border-color:#BBBBBB; border-width:1px; padding-left:4ps; color:gray; font-size: 1em; }
	</style>

	{{.JS}}

</head>


<body>

	<span style="color:gray; font-weight:bold; font-size:1.5em"> {{.Data.Title}} &nbsp; &nbsp; </span> {{.UpdateButton ""}} {{.UpdateBox "live"}} &nbsp; &nbsp; {{.ErrorBox}} <br/>
	<hr/>

</body>
</html>
`
