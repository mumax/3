//+build ignore

package main

import (
	. "."
	"fmt"
	"time"
)

func main() {
	v := NewServer(testtempl)
	t := new(test)

	v.Add("time", time.Now)
	v.Add("hit me", t.HitMe)

	a := &alpha{"it works"}
	v.Add("alpha", a)

	v.ListenAndServe(":7070")
}

type alpha struct{ x string }

func (a *alpha) Get() interface{}  { fmt.Println("get alpha"); return a.x }
func (a *alpha) Set(v interface{}) { a.x = v.(string) + "*"; fmt.Println("set alpha") }

type test struct{ hits int }

func (t *test) SayHello() string { return "Hello world wide web!" }
func (t *test) HitMe()           { t.hits++; fmt.Println("got hit") }
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

	It's now <b> {{.Label "time"}} </b><br/><br/>
	{{.Button "hit me"}}

	{{.TextBox "alpha"}}

	<hr/>
	
	{{.AutoRefreshBox}}

</body>
</html>
`
