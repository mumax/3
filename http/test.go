package main

import (
	"log"
	"os"
	"time"
)

func main() {
	log.SetFlags(0)
	testdata := &testt{}
	v := NewView(testdata, testtempl)
	v.Render(os.Stdout)
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

	{{.Static "SayHello"}}
	{{.Dynamic "Time"}}

</body>
</html>
`
