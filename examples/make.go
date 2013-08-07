/*
Tool to quickly plot mx3 data tables using gnuplot.

Usage

Run
	mx3-plot datatable.txt
and SVG graphs will appear in that directory.
*/
package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"sort"
	"strings"
	"text/template"
)

var flag_vet = flag.Bool("vet", false, "only vet source files, don't run them")

func main() {
	flag.Parse()

	// read template
	b, err := ioutil.ReadFile("template.html")
	check(err)
	replaceInRaw(b, '\n', '@') // hack to allow raw strings spanning multi lines
	templ := template.Must(template.New("guide").Parse(string(b)))

	// output file
	f, err2 := os.OpenFile("examples.html", os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	check(err2)

	// execute!
	state := &State{}
	check(templ.Execute(f, state))
}

type State struct {
	count int
}

func (s *State) Example(in string) string {
	s.count++

	// extract example source
	in = strings.Replace(in, "@", "\n", -1) // undo raw string hack
	in = strings.Trim(in, "\n")

	// exec input file
	check(ioutil.WriteFile(s.infile(), []byte(in), 0666))
	arg := "-s"
	if *flag_vet {
		arg = "-vet"
	}
	cmd("mx3", "-f", arg, s.infile())

	return `<pre>` + template.HTMLEscapeString(in) + `</pre>`
}

func (s *State) Img(fname string) string {
	cmd("mx3-convert", "-png", s.outfile()+"/"+fname+".dump")
	pngfile := s.outfile() + "/" + fname + ".png"
	return fmt.Sprintf(`
<figure style="float:left">
	<img src="%v"/>
	<figcaption> %v </figcaption>
</figure>`, pngfile, fname)
}

func (s *State) Include(fname string) string {
	b, err := ioutil.ReadFile(fname)
	check(err)
	return string(b)
}

func (s *State) Output() string {
	out := `<h3>output</h3> `

	dir, err := os.Open(s.outfile())
	check(err)
	files, err2 := dir.Readdirnames(-1)
	check(err2)
	sort.Strings(files)
	for _, f := range files {
		if path.Ext(f) == ".dump" {
			out += s.Img(f[:len(f)-len(".dump")])
		}
	}
	out += `<br style="clear:both"/> `

	for _, f := range files {
		if path.Ext(f) == ".txt" {
			cmd("mx3-plot", s.outfile()+"/"+f)
		}
	}

	dir, err = os.Open(s.outfile())
	check(err)
	files, err2 = dir.Readdirnames(-1)
	check(err2)
	sort.Strings(files)
	for _, f := range files {
		if path.Ext(f) == ".svg" {
			src := s.outfile() + "/" + f
			out += fmt.Sprintf(`
<figure>
	<img src="%v"/>
	<figcaption> %v </figcaption>
</figure>`, src, f)
		}
	}
	return out
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
		log.Panic(err)
	}
}
