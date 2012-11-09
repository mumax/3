package main

// Author: Arne Vansteenkiste, Mykola Dvornik

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/dump"
	"flag"
	"fmt"
	"io"
	"os"
	"path"
)

var (
	flag_crc         = flag.Bool("crc", true, "Generate/check CRC checksums")
	flag_onefile     = flag.Bool("onefile", false, "Using one file for output")
	flag_show        = flag.Bool("show", false, "Human-readible output to stdout")
	flag_format      = flag.String("f", "%v", "Printf format string")
	flag_png         = flag.Bool("png", false, "PNG output")
	flag_jpeg        = flag.Bool("jpeg", false, "JPEG output")
	flag_gnuplot     = flag.Bool("gplot", false, "Gnuplot-compatible output")
	flag_gnuplotgzip = flag.Bool("gplotgzip", false, "Gzip'ed 'Gnuplot-compatible output")
	flag_omf         = flag.String("omf", "", `"text" or "binary" OMF output`)
	flag_vtk         = flag.String("vtk", "", `"ascii" or "binary" VTK output`)
	flag_min         = flag.String("min", "auto", `Minimum of color scale: "auto" or value.`)
	flag_max         = flag.String("max", "auto", `Maximum of color scale: "auto" or value.`)
	flag_normalize   = flag.Bool("normalize", false, `Normalize vector data to unit length`)
)

const (
	X = iota
	Y
	Z
)

func main() {
	flag.Parse()
	core.LOG = false

	if flag.NArg() == 0 {
		read(os.Stdin, "")
	}
	for _, arg := range flag.Args() {
		f, err := os.Open(arg)
		core.Fatal(err)
		read(f, arg)
		f.Close()
	}
}

func read(in io.Reader, name string) {
	r := dump.NewReader(in, *flag_crc)
	err := r.Read()
	i := 0
	ext := path.Ext(name)
	woext := noExt(name)
	for err != io.EOF {
		core.Fatal(err)
		tname := name
		if !(*flag_onefile) {
			num := fmt.Sprintf("%06d", i)
			tname = woext + num + ext
		}
		process(&r.Frame, tname)
		err = r.Read()
		i = i + 1
	}
}

func process(f *dump.Frame, name string) {
	preprocess(f)

	haveOutput := false

	if *flag_jpeg {
		dumpImage(f, noExt(name)+".jpg")
		haveOutput = true
	}

	if *flag_png {
		dumpImage(f, noExt(name)+".png")
		haveOutput = true
	}

	if *flag_gnuplot {
		dumpGnuplot(f, noExt(name)+".gplot")
		haveOutput = true
	}

	if *flag_gnuplotgzip {
		dumpGnuplotGZip(f, noExt(name)+".gplot.gz")
		haveOutput = true
	}

	if *flag_omf != "" {
		dumpOmf(noExt(name)+".omf", f, *flag_omf)
		haveOutput = true
	}

	if *flag_vtk != "" {
		dumpVTK(noExt(name)+".vtk", f, *flag_vtk)
		haveOutput = true
	}

	if !haveOutput || *flag_show {
		f.Fprintf(os.Stdout, *flag_format)
		haveOutput = true
	}
}

func noExt(file string) string {
	ext := path.Ext(file)
	return file[:len(file)-len(ext)]
}

func preprocess(f *dump.Frame) {
	if *flag_normalize {
		normalize(f)
	}
}

// Transforms the index between user and program space, unless it is a scalar:
//	X  <-> Z
//	Y  <-> Y
//	Z  <-> X
//	XX <-> ZZ
//	YY <-> YY
//	ZZ <-> XX
//	YZ <-> XY
//	XZ <-> XZ
//	XY <-> YZ 
func SwapIndex(index, dim int) int {
	switch dim {
	default:
		panic(fmt.Errorf("dim=%v", dim))
	case 1:
		return index
	case 3:
		return [3]int{Z, Y, X}[index]
	case 6:
		return [6]int{ZZ, YY, XX, XY, XZ, YZ}[index]
	case 9:
		return [9]int{ZZ, YY, XX, YX, ZX, ZY, XY, XZ, YZ}[index]
	}
	return -1 // silence 6g
}

// Linear indices for matrix components.
// E.g.: matrix[Y][Z] is stored as list[YZ]
const (
	XX = 0
	YY = 1
	ZZ = 2
	YZ = 3
	XZ = 4
	XY = 5
	ZY = 6
	ZX = 7
	YX = 8
)
