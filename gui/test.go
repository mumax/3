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
		body   { margin: 20px; font-family: Ubuntu, Arial, sans-serif; }
		.ErrorBox { color: red; font-weight: bold; } 
	</style>
	{{.JS}}
</head>

<body onload=refresh()>

	<p> {{.ErrorBox}} </p>

	{{.Static "SayHello"}} <br/><br/>

	It's now <b> {{.Label "Time"}} </b><br/><br/>

	{{.Button "HitMe"}}
	You hit me {{.Label "HitCount"}} fucking times! <br/><br/>

</body>
</html>
`
