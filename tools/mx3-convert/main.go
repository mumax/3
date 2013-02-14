/*
 This tool converts binary output files to various formats and allows basic manipulations like crop, rescale...
 Usage:
 	mx3-convert [flags] files
 For a overview of flags, run:
 	mx3-convert -help
 Example: convert all .dump files to PNG:
 	mx3-convert -png *.dump
 Example: resize data to a 1 x 32 x 32 mesh, normalize vectors to unit length and convert the result to OOMMF binary output:
 	mx3-convert -resize 1x32x32 -normalize -omf binary file.dump
*/
package main

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"flag"
	"log"
	"os"
	"runtime"
	"sync"
)

var (
	flag_gzip      = flag.Bool("gzip", false, "GZIP the output")
	flag_show      = flag.Bool("show", false, "Human-readible output to stdout")
	flag_format    = flag.String("f", "%v", "Printf format string")
	flag_png       = flag.Bool("png", false, "PNG output")
	flag_jpeg      = flag.Bool("jpg", false, "JPEG output")
	flag_gnuplot   = flag.Bool("gplot", false, "Gnuplot-compatible output")
	flag_omf       = flag.String("omf", "", `"text" or "binary" OMF output`)
	flag_vtk       = flag.String("vtk", "", `"ascii" or "binary" VTK output`)
	flag_dump      = flag.Bool("dump", false, `output in dump format`)
	flag_min       = flag.String("min", "auto", `Minimum of color scale: "auto" or value.`)
	flag_max       = flag.String("max", "auto", `Maximum of color scale: "auto" or value.`)
	flag_normalize = flag.Bool("normalize", false, `Normalize vector data to unit length`)
	flag_normpeak  = flag.Bool("normpeak", false, `Scale vector data, maximum to unit length`)
	flag_o         = flag.String("o", "", "Set output file name format. %v is replaced by input file. E.g.: scaled_%v")
	flag_resize    = flag.String("resize", "", "Resize. E.g.: 4x128x128")
	//flag_force     = flag.Bool("f", false, "Force overwrite of existing files")
	// TODO: crop, component
)

var que chan task
var wg sync.WaitGroup

type task struct {
	*data.Slice
	fname string
}

func main() {
	log.SetFlags(0)
	flag.Parse()
	if flag.NArg() == 0 {
		log.Fatal("no input files")
	}

	// start many worker goroutines taking tasks from que
	runtime.GOMAXPROCS(runtime.NumCPU())
	ncpu := runtime.GOMAXPROCS(-1)
	que = make(chan task, ncpu)
	if ncpu == 0 {
		ncpu = 1
	}
	for i := 0; i < ncpu; i++ {
		go work()
	}

	// read all input files and put them in the task que
	for _, fname := range flag.Args() {
		slice, err := data.ReadSliceFile(fname)
		if err != nil {
			log.Println(err)
			continue
		}
		wg.Add(1)
		que <- task{slice, util.NoExt(fname)}
	}

	// wait for work to finish
	wg.Wait()
}

func work() {
	for task := range que {
		log.Println(task.fname)
		process(task.Slice, task.fname)
		wg.Done()
	}
}

func open(fname string) *os.File {
	f, err := os.OpenFile(fname, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	util.FatalErr(err)
	return f
}

func process(f *data.Slice, name string) {
	preprocess(f)

	haveOutput := false

	if *flag_jpeg {
		out := open(name + ".jpg")
		defer out.Close()
		dumpJPEG(out, f)
		haveOutput = true
	}

	if *flag_png {
		out := open(name + ".png")
		defer out.Close()
		dumpPNG(out, f)
		haveOutput = true
	}

	if *flag_gnuplot {
		out := open(name + ".gplot")
		defer out.Close()
		dumpGnuplot(out, f)
		haveOutput = true
	}

	if *flag_omf != "" {
		out := open(name + ".omf")
		defer out.Close()
		dumpOmf(out, f, *flag_omf)
		haveOutput = true
	}

	if *flag_vtk != "" {
		out := open(name + ".vtk")
		defer out.Close()
		dumpVTK(out, f, *flag_vtk)
		haveOutput = true
	}

	//	if *flag_dump {
	//		dumpDump(name+".dump", f)
	//		haveOutput = true
	//	}

	if !haveOutput {
		log.Fatal("need to specifiy at least one output flag")
	}
	//	if !haveOutput || *flag_show {
	//		f.Fprintf(os.Stdout, *flag_format)
	//		haveOutput = true
	//	}

}

func preprocess(f *data.Slice) {
	if *flag_normalize {
		normalize(f, 1)
	}
	if *flag_normpeak {
		normpeak(f)
	}
	if *flag_resize != "" {
		resize(f, *flag_resize)
	}
	//if *flag_scale != 1{
	//	rescale(f, *flag_scale)
	//}
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
func swapIndex(index, dim int) int {
	switch dim {
	default:
		log.Panicf("dim=%v", dim)
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
	X  = 0
	Y  = 1
	Z  = 2
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
