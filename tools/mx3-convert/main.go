/*
mx3-convert converts mx3 output files and .omf files to various formats and images.
It also provides basic manipulations like data rescale etc.


Usage

Command-line flags must always preceed the input files:
	mx3-convert [flags] files
For a overview of flags, run:
	mx3-convert -help
Example: convert all .dump files to PNG:
	mx3-convert -png *.dump
Example: resize data to a 32 x 32 x 1 mesh, normalize vectors to unit length and convert the result to OOMMF binary output:
	mx3-convert -resize 32x32x1 -normalize -omf binary file.dump
Example: convert all .omf files to VTK binary saving only the X component. Also output to JPEG in the meanwhile:
	mx3-convert -comp 0 -vtk binary -jpg *.omf
Example: convet .omf files to .dump, so they can be used as input for mx3 simulations:
	mx3-convert -dump *.omf

Output file names are automatically assigned.
*/
package main

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"flag"
	"log"
	"os"
	"path"
	"runtime"
	"sync"
)

var (
	flag_comp      = flag.Int("comp", -1, "Select a component of vector data. 0=x, 1=y, ...")
	flag_show      = flag.Bool("show", false, "Human-readible output to stdout")
	flag_format    = flag.String("f", "%v", "Printf format string")
	flag_png       = flag.Bool("png", false, "PNG output")
	flag_jpeg      = flag.Bool("jpg", false, "JPEG output")
	flag_gnuplot   = flag.Bool("gplot", false, "Gnuplot-compatible output")
	flag_omf       = flag.String("omf", "", `"text" or "binary" OMF (OVF1) output`)
	flag_ovf1      = flag.String("ovf1", "", `"text" or "binary" OVF1 output`)
	flag_ovf2      = flag.String("ovf2", "", `"text" or "binary" OVF2 output`)
	flag_vtk       = flag.String("vtk", "", `"ascii" or "binary" VTK output`)
	flag_dump      = flag.Bool("dump", false, `output in dump format`)
	flag_min       = flag.String("min", "auto", `Minimum of color scale: "auto" or value.`)
	flag_max       = flag.String("max", "auto", `Maximum of color scale: "auto" or value.`)
	flag_normalize = flag.Bool("normalize", false, `Normalize vector data to unit length`)
	flag_normpeak  = flag.Bool("normpeak", false, `Scale vector data, maximum to unit length`)
	flag_resize    = flag.String("resize", "", "Resize. E.g.: 4x128x128")
)

var que chan task
var wg sync.WaitGroup

type task struct {
	*data.Slice
	time  float64
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
		log.Println(fname)
		var slice *data.Slice
		var time float64
		var err error

		if path.Ext(fname) == ".omf" {
			slice, time, err = ReadOMF(fname)
		} else {
			slice, time, err = data.ReadFile(fname)
		}
		if err != nil {
			log.Println(err)
			continue
		}
		wg.Add(1)
		que <- task{slice, time, util.NoExt(fname)}
	}

	// wait for work to finish
	wg.Wait()
}

func work() {
	for task := range que {
		process(task.Slice, task.time, task.fname)
		wg.Done()
	}
}

func open(fname string) *os.File {
	f, err := os.OpenFile(fname, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	util.FatalErr(err)
	return f
}

func process(f *data.Slice, time float64, name string) {
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

	if *flag_ovf1 != "" {
		out := open(name + ".ovf")
		defer out.Close()
		dumpOmf(out, f, *flag_ovf1)
		haveOutput = true
	}

	if *flag_ovf2 != "" {
		out := open(name + ".ovf")
		defer out.Close()
		dumpOvf2(out, f, *flag_ovf2, time)
		haveOutput = true
	}

	if *flag_vtk != "" {
		out := open(name + ".vts") // vts is the official extension for VTK files containing StructuredGrid data
		defer out.Close()
		dumpVTK(out, f, *flag_vtk)
		haveOutput = true
	}

	if *flag_dump {
		data.MustWriteFile(name+".dump", f, time)
		haveOutput = true
	}

	if !haveOutput || *flag_show {
		// TODO: header
		util.Fprintf(os.Stdout, *flag_format, f.Tensors())
		haveOutput = true
	}

}

func preprocess(f *data.Slice) {
	if *flag_normalize {
		normalize(f, 1)
	}
	if *flag_normpeak {
		normpeak(f)
	}
	if *flag_comp != -1 {
		*f = *f.Comp(util.SwapIndex(*flag_comp, f.NComp()))
	}
	if *flag_resize != "" {
		resize(f, *flag_resize)
	}
}
