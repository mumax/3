//+build ignore

package main

import (
	. "."
	//"fmt"
	"net/http"

//"time"
)

func main() {
	p := NewPage(testtempl, nil)
	http.Handle("/", p)
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		log.Fatal(err)
	}
}

const testtempl = `
<html>

<head>
	<style type="text/css">
		body      { margin: 20px; font-family: Ubuntu, Arial, sans-serif; }
		hr        { border-style: none; border-top: 1px solid #CCCCCC; }
		.ErrorBox { color: red; font-weight: bold; } 
		.TextBox  { border:solid; border-color:#BBBBBB; border-width:1px; padding-left:4px;}
	</style>
	{{.JS}}
</head>

<body>

	<h1> GUI test </h1>
	<p> {{.ErrorBox}} </p>
	<p> <button onclick="refresh();"> &#x21bb; </button> </p>
	<hr/>


	<hr/>
</body>
</html>
`
