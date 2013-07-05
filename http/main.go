package main

import (
	"log"
	"os"
)

func main() {
	log.SetFlags(0)
	testdata := &struct{}{}
	v := NewView(testdata, testtempl)
	v.Render(os.Stdout)
}

const testtempl = `
<html>
<head>
{{.JS}}
</head>
<body>
</body>
</html>
`
