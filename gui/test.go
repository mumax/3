//+build ignore

package main

import (
	. "."
	"time"
)

func main() {
	testdata := &test{}
	v := NewServer(testdata, testtempl)
	v.ListenAndServe(":7070")
}

type test struct{ hits int }

func (t *test) SayHello() string { return "Hello world wide web!" }
func (t *test) HitMe()           { t.hits++ }
func (t *test) HitCount() int    { return t.hits }
func (t *test) Time() time.Time  { return time.Now() }

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

<body onload=refresh()>

	<h1> GUI test </h1>
	<p> {{.ErrorBox}} </p>
	<hr/>

	{{.Static "SayHello"}} <br/><br/>

	It's now <b> {{.Label "Time"}} </b><br/><br/>

	{{.Button "HitMe"}}
	You hit me {{.Label "HitCount"}} fucking times! <br/><br/>

	{{.TextBox "TextMe"}}

	<hr/>

	{{.AutoRefreshBox}}

</body>
</html>
`
