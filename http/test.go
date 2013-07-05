package main

import (
	"log"
	"time"
)

func main() {
	log.SetFlags(0)
	testdata := &testt{}
	v := NewView(testdata, testtempl)
	v.ListenAndServe(":7070")
}

type testt struct{ hits int }

func (t *testt) SayHello() string { return "Hello world wide web!" }
func (t *testt) HitMe()           { t.hits++ }
func (t *testt) HitCount() int    { return t.hits }
func (t *testt) Time() time.Time    { return time.Now() }

const testtempl = `
<html>

<head>
	<style media="all" type="text/css">
		body { margin: 20px; font-family: Ubuntu, Arial, sans-serif; font-size: 14px; color: #444444}
	</style>

{{.JS}}
</head>

<body onload=refresh()>

	<p><b>{{.Err}}</b></p>

	{{.Static "SayHello"}} <br/><br/>

	It's now <b> {{.Label "Time"}} </b><br/><br/>

	{{.Button "HitMe"}}
	You hit me {{.Label "HitCount"}} fucking times! <br/><br/>

</body>
</html>
`
