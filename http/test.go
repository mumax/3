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

type testt struct {
}

func (t *testt) SayHello() string {
	return "<Hello world wide web!>"
}

func (t *testt) Time() time.Time {
	return time.Now()
}

const testtempl = `
<html>

<head>
{{.JS}}
</head>

<body>

	<p><b>{{.Err}}</b></p>

	{{.Static "SayHello"}}

	{{.Dynamic "Time"}}

</body>
</html>
`
