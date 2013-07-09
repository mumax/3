//+build ignore

package main

import (
	. "."
	"time"
)

func main() {
	v := NewServer(testtempl)
	t := new(test)

	v.Add("time", time.Now)
	v.Add("hit me", t.HitMe)

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

	It's now <b> {{.Label "time"}} </b><br/><br/>

	<hr/>
	
	{{.AutoRefreshBox}}

</body>
</html>
`
