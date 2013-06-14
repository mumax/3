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

	// extract example source
	in = strings.Replace(in, "@", "\n", -1) // undo raw string hack
	in = strings.Trim(in, "\n")

	// exec input file
	check(ioutil.WriteFile(s.infile(), []byte(in), 0666))
	cmd("mx3", "-f", s.infile())

	s.count++
	return `<pre>` + in + `</pre>`
}

func (s *State) infile() string {
	return fmt.Sprintf("example%v.txt", s.count)
}

func (s *State) outfile() string {
	return fmt.Sprintf("example%v.out", s.count)
}

func cmd(cmd string, args ...string) {
	out, err := exec.Command(cmd, args...).CombinedOutput()
	os.Stdout.Write(out)
	check(err)

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
