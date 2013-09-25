/*
The mumax3-plot utility uses gnuplot to automatically plot mumax3 data tables.
	mumax3-plot table.txt
Creates graphs of all columns as .svg files.
*/
package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path"
	"strings"
)

func main() {
	log.SetFlags(0)
	flag.Parse()

	for _, f := range flag.Args() {
		plotFile(f)
	}
}

func plotFile(fname string) {

	hdr := readHeader(fname)

	// quantities grouped by vector
	Qs := []*Q{&Q{[]string{"t"}, "s", []int{1}}}
	prev := Qs[0]

	quants := strings.Split(hdr, "\t")
	for i := 1; i < len(quants); i++ {
		spl := strings.Split(quants[i], " ")
		name := spl[0]
		unit := spl[1]
		if unit == "()" {
			unit = ""
		}

		if name[:len(name)-1] == prev.name[0][:len(prev.name[0])-1] {
			prev.cols = append(prev.cols, i+1)
			prev.name = append(prev.name, name)
		} else {
			n := &Q{[]string{name}, unit, []int{i + 1}}
			Qs = append(Qs, n)
			prev = n
		}
	}
	log.Println(Qs)

	for i := 1; i < len(Qs); i++ {
		makePlot(fname, Qs[i])
	}
}

func makePlot(fname string, q *Q) {
	term := "svg"
	outf := path.Dir(fname) + "/" + q.vecname()
	cmd := fmt.Sprintf(`set term %v size 400 300 fsize 10; set output "%v.%v";`, term, outf, term)
	cmd += fmt.Sprintf(`set xlabel "t(ns)";`)

	cmd += fmt.Sprintf(`set ylabel "%v %v";`, q.vecname(), q.unit)
	cmd += fmt.Sprint(`set format y "%g";`)
	cmd += fmt.Sprint(`plot "`, fname, `" u ($1*1e9):`, q.cols[0], ` w li title "`, q.name[0], `"`)
	for i := 1; i < len(q.cols); i++ {
		cmd += fmt.Sprint(`, "`, fname, `" u ($1*1e9):`, q.cols[i], ` w li title "`, q.name[i], `"`)
	}
	cmd += "; set output;"

	out, err := exec.Command("gnuplot", "-e", cmd).CombinedOutput()
	os.Stderr.Write(out)
	check(err)
}

type Q struct {
	name []string
	unit string
	cols []int
}

func (q *Q) String() string { return fmt.Sprint(q.name, "(", q.unit, ")", q.cols) }

func (q *Q) vecname() string {
	if len(q.cols) > 1 {
		return q.name[0][:len(q.name[0])-1]
	} else {
		return q.name[0]
	}
}

func readHeader(fname string) string {
	f, err := os.Open(fname)
	check(err)
	defer f.Close()
	in := bufio.NewReader(f)
	hdrBytes, _, err2 := in.ReadLine()
	check(err2)
	hdr := string(hdrBytes)
	if hdr[0] != '#' {
		log.Fatal("invalid table header:", hdr)
	}
	hdr = hdr[2:]
	return hdr
}

func check(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
