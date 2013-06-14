package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"strings"
	"text/template"
)

func main() {
	// read template
	b, err := ioutil.ReadFile("template.html")
	check(err)
	replaceInRaw(b, '\n', '@') // hack to allow raw strings spanning multi lines
	templ := template.Must(template.New("guide").Parse(string(b)))

	// output file
	f, err2 := os.OpenFile("guide.html", os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	check(err2)

	// execute!
	state := &State{}
	check(templ.Execute(f, state))
}

type State struct {
	count int
}

func (s *State) Example(in string) string {

	in = strings.Replace(in, "@", "\n", -1) // undo raw string hack
	in = strings.Trim(in, "\n")

	infile := fmt.Sprintf("example%v.txt", s.count)
	check(ioutil.WriteFile(infile, []byte(in), 0666))
	out, err := exec.Command("mx3", "-f", infile).CombinedOutput()
	os.Stdout.Write(out)
	check(err)

	s.count++
	return `<pre>` + in + `</pre>`

}

func replaceInRaw(bytes []byte, old, new byte) {
	inraw := false
	for i, b := range bytes {
		if b == '`' {
			inraw = !inraw
		}
		if inraw && b == old {
			bytes[i] = new
		}
	}
}

func check(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
