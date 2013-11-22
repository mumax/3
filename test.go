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
	p.Attr("attrtest", "innerHTML", "innerHTML works")
	p.Attr("range", "max", 200)
	p.Disable("button2", true)
	p.Disable("text3", true)
	p.Disable("button", true)
	p.Disable("button", false)
	p.OnUpdate(func() {
		p.Set("time", time.Now().Format(time.ANSIC))
	})
	p.OnEvent("button", func() {
		p.Set("button", p.StringValue("button")+" again")
	})
	p.OnEvent("text", func() {
		p.Set("text2", p.StringValue("text"))
	})
	p.OnEvent("text2", func() {
		p.Set("text", p.StringValue("text2"))
	})
	p.OnEvent("range", func() {
		p.Set("rangeEcho", p.Value("range"))
	})
	p.OnEvent("check", func() {
		p.Disable("check", p.BoolValue("check"))
	})
	p.OnEvent("select", func() {
		p.Set("selectEcho", p.Value("select"))
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
	<p>{{.Button "button2" "don't click" }}	{{.Button "button" "click me" }} </p>
	<p>{{.TextBox "text3" "don't type" }} {{.TextBox "text2" "" "placeholder='type here'"}} {{.TextBox "text" "echo here" }} </p>
	<p>{{.TextArea "texta" 8 64 ""}} </p>
	<p>{{.Range "range" 0 100 50}} {{.Span "rangeEcho" 50}} </p>
	<p>{{.Checkbox "check" "checked" true}} </p>
	<p>{{.Select "select" "a" "a" "b" "c"}} {{.Select "selectEcho" "a" "a" "b" "c"}} </p>
	<hr/>
</body>
</html>
`
