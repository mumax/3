package main

import (
	"log"
)

func main() {
	log.SetFlags(0)
	testdata := &testt{}
	v := NewView(testdata, testtempl)
	v.ListenAndServe(":7070")
}

type testt struct{ hits int }

func (t *testt) SayHello() string { return "<Hello world wide web!>" }
func (t *testt) HitMe()           { t.hits++ }
func (t *testt) HitCount() int    { return t.hits }

const testtempl = `
<html>

<head>
{{.JS}}
</head>

<body>

	<p><b>{{.Err}}</b></p>

	{{.Static "SayHello"}}
	{{.Label "HitCount"}}
	{{.Button "HitMe"}}


</body>
</html>
`
