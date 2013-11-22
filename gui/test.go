//+build ignore

package main

import (
	. "."
	"log"
	"net/http"
	"time"
)

func main() {
	p := NewPage(testtempl, nil)
	p.Attr("attrtest", "innerHTML", "it works")
	p.OnUpdate(func() {
		p.Set("time", time.Now().Format(time.ANSIC))
	})
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
	<p> {{.UpdateButton ""}} {{.UpdateBox "live"}} {{.ErrorBox}} </p>
	<hr/>
	
	<p>{{.Span "static" "static span" "style=color:blue"}} </p>
	<p>{{.Span "time" "time" }} </p>
	<p>{{.Span "attrtest" "" }} </p>

	<hr/>
</body>
</html>
`
