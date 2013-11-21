//+build ignore

package main

import (
	. "."
	"fmt"
	"log"
	//"fmt"
	"net/http"
	//"time"
)

func main() {
	p := NewPage(testtempl, nil, onrefresh)
	http.Handle("/", p)
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		log.Fatal(err)
	}
}

func onrefresh() {
	fmt.Println("*")
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
	<p> {{.RefreshButton}} {{.RefreshBox}}
	<hr/>
	
	{{.Span "time" "time flies" "style=color:blue"}}

	<hr/>
</body>
</html>
`
