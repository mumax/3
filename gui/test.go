//+build ignore

package main

import (
	. "."
	"fmt"
	"log"
	"net/http"
)

func main() {
	p := NewPage(testtempl, nil)
	p.OnUpdate(func() {
		fmt.Println("*")
		//p.Set("time", time.Now)
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
	<p> {{.ErrorBox}} </p>
	<p> {{.UpdateButton}} {{.UpdateBox}}
	<hr/>
	
	{{.Span "static" "static span" "style=color:blue"}}
	{{.Span "time" "time" }}

	<hr/>
</body>
</html>
`
