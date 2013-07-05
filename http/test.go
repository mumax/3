package main

import (
	"log"
	"os"
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

const testtempl = `
<html>
<head>
{{.JS}}
</head>
<body>

	{{.Static "SayHello"}}

</body>
</html>
`
