package main

import (
	mxio "code.google.com/p/mx3/io"
	"code.google.com/p/mx3/mx"
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

func init() {
	runtime.GOMAXPROCS(runtime.NumCPU())
}

var que = make(chan task, 2)
var wg sync.WaitGroup

type task struct {
	*mx.Slice
	fname string
}

func main() {
	flag.Parse()
	if flag.NArg() == 0 {
		log.Fatal("no input files")
	}

	for _, fname := range flag.Args() {
		slice, err := mxio.ReadSliceFile(fname)
		if err != nil {
			log.Println(err)
			continue
		}
		wg.Add(1)
		que <- task{slice, util.NoExt(fname)}
	}
	wg.Wait()
}

//func process(f *dump.Frame, name string) {
//	preprocess(f)
//
//	haveOutput := false
//
//	if *flag_jpeg {
//		dumpImage(f, core.NoExt(name)+".jpg")
//		haveOutput = true
//	}
//
//	if *flag_png {
//		dumpImage(f, core.NoExt(name)+".png")
//		haveOutput = true
//	}
//
//	if *flag_gnuplot {
//		dumpGnuplot(f, core.NoExt(name)+".gplot")
//		haveOutput = true
//	}
//
//	if *flag_omf != "" {
//		dumpOmf(core.NoExt(name)+".omf", f, *flag_omf)
//		haveOutput = true
//	}
//
//	if *flag_vtk != "" {
//		dumpVTK(core.NoExt(name)+".vtk", f, *flag_vtk)
//		haveOutput = true
//	}
//
//	if *flag_dump {
//		dumpDump(core.NoExt(name)+".dump", f)
//		haveOutput = true
//	}
//
//	if !haveOutput || *flag_show {
//		f.Fprintf(os.Stdout, *flag_format)
//		haveOutput = true
//	}
//}
//
//func preprocess(f *dump.Frame) {
//	if *flag_normalize {
//		normalize(f, 1)
//	}
//	if *flag_normpeak {
//		normpeak(f)
//	}
//	//if *flag_scale != 1{
//	//	rescale(f, *flag_scale)
//	//}
//}
//
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
