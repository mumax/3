/*
 This tool converts binary output files to various formats and allows basic manipulations like crop, rescale...
 Usage:
 	mx3-convert [flags] files
 For a overview of flags, run:
 	mx3-convert -help
 Example: convert all .dump files to PNG:
 	mx3-convert -png *.dump
*/
package main

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"flag"
	"log"
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

var que = make(chan task, 2)
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
	ncpu := runtime.NumCPU() - 1
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

func process(f *data.Slice, name string) {
	preprocess(f)

	haveOutput := false

	if *flag_jpeg {
		dumpImage(f, name+".jpg")
		haveOutput = true
	}

	if *flag_png {
		dumpImage(f, name+".png")
		haveOutput = true
	}

	if *flag_gnuplot {
		dumpGnuplot(f, name+".gplot")
		haveOutput = true
	}

	if *flag_omf != "" {
		dumpOmf(name+".omf", f, *flag_omf)
		haveOutput = true
	}

	if *flag_vtk != "" {
		dumpVTK(name+".vtk", f, *flag_vtk)
		haveOutput = true
	}

	if *flag_dump {
		dumpDump(name+".dump", f)
		haveOutput = true
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
	//if *flag_scale != 1{
	//	rescale(f, *flag_scale)
	//}
}

//// Transforms the index between user and program space, unless it is a scalar:
////	X  <-> Z
////	Y  <-> Y
////	Z  <-> X
////	XX <-> ZZ
////	YY <-> YY
////	ZZ <-> XX
////	YZ <-> XY
////	XZ <-> XZ
////	XY <-> YZ
//func SwapIndex(index, dim int) int {
//	switch dim {
//	default:
//		panic(fmt.Errorf("dim=%v", dim))
//	case 1:
//		return index
//	case 3:
//		return [3]int{Z, Y, X}[index]
//	case 6:
//		return [6]int{ZZ, YY, XX, XY, XZ, YZ}[index]
//	case 9:
//		return [9]int{ZZ, YY, XX, YX, ZX, ZY, XY, XZ, YZ}[index]
//	}
//	return -1 // silence 6g
//}
//
//// Linear indices for matrix components.
//// E.g.: matrix[Y][Z] is stored as list[YZ]
//const (
//	XX = 0
//	YY = 1
//	ZZ = 2
//	YZ = 3
//	XZ = 4
//	XY = 5
//	ZY = 6
//	ZX = 7
//	YX = 8
//)
