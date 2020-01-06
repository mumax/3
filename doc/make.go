package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"regexp"
	"sort"
	"strings"
	"text/template"
)

var flag_vet = flag.Bool("vet", false, "only vet source files, don't run them")
var flag_api = flag.Bool("api", false, "only generate API")

func main() {
	flag.Parse()

	buildAPI()

	// read template
	b, err := ioutil.ReadFile("template.html")
	check(err)
	replaceInRaw(b, '\n', '@') // hack to allow raw strings spanning multi lines
	templ := template.Must(template.New("guide").Parse(string(b)))

	// output file
	f, err2 := os.OpenFile("examples.html", os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	check(err2)

	// execute!
	if !*flag_api {
		state := &State{}
		check(templ.Execute(f, state))
	}

	renderAPI()
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
	arg := "-v"
	if *flag_vet {
		arg = "-vet"
	}
	cmd("mumax3", "-cache", "/tmp", arg, s.infile())

	recordExamples(in, s.count)

	return `<a id=example` + fmt.Sprint(s.count) + `></a><pre>` + template.HTMLEscapeString(in) + `</pre>`
}

var api_examples = make(map[string][]int)

func recordExamples(input string, num int) {
	in := strings.ToLower(input)
	for k, _ := range api_ident {
		if ok, _ := regexp.MatchString(k, in); ok {
			api_examples[k] = append(api_examples[k], num)
		}
	}
}

func (s *State) Img(fname string) string {
	cmd("mumax3-convert", "-png", "-arrows", "16", s.outfile()+"/"+fname+".ovf")
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
		if path.Ext(f) == ".ovf" {
			out += s.Img(f[:len(f)-len(".ovf")])
		}
	}
	out += `<br style="clear:both"/> `

	for _, f := range files {
		if f == "table.txt" {
			cmd("mumax3-plot", s.outfile()+"/"+f)
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

// State.output gives a nice otuput for all examples except for the
// hysteresis example. State.OutputHysteresis is the custom output function
// for the hysteresis example.
func (s *State) OutputHysteresis() string {
	tableName := fmt.Sprintf(`%s/table.txt`, s.outfile())
	figureName := fmt.Sprintf(`%s/hysteresis.svg`, s.outfile())

	gnuplotCmd := `set term svg noenhanced size 400 300 font 'Arial,10';`
	gnuplotCmd += fmt.Sprintf(`set output "%s";`, figureName)
	gnuplotCmd += `set xlabel "B_ext(T)";`
	gnuplotCmd += `set ylabel "m_x";`
	gnuplotCmd += fmt.Sprintf(`plot "%s" u 5:2 w lp notitle;`, tableName)
	gnuplotCmd += "set output;"

	gnuplotOut, err := exec.Command("gnuplot", "-e", gnuplotCmd).CombinedOutput()
	os.Stderr.Write(gnuplotOut)
	check(err)

	out := fmt.Sprintf(`
<h3>output</h3>
<figure>
	<img src="%v"/>
</figure>`, figureName)

	return out
}

func (s *State) infile() string {
	return fmt.Sprintf("example%v.mx3", s.count)
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
