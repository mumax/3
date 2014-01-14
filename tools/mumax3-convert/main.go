/*
mumax3-convert converts mumax3 output files and .ovf files to various formats and images.
It also provides basic manipulations like data rescale etc.


Usage

Command-line flags must always preceed the input files:
	mumax3-convert [flags] files
For a overview of flags, run:
	mumax3-convert -help
Example: convert all .dump files to PNG:
	mumax3-convert -png *.dump
Example: resize data to a 32 x 32 x 1 mesh, normalize vectors to unit length and convert the result to OOMMF binary output:
	mumax3-convert -resize 32x32x1 -normalize -ovf binary file.dump
Example: convert all .ovf files to VTK binary saving only the X component. Also output to JPEG in the meanwhile:
	mumax3-convert -comp 0 -vtk binary -jpg *.ovf
Example: convert .ovf files to .dump, so they can be used as input for mumax3 simulations:
	mumax3-convert -dump *.ovf
Example: cut out a piece of the data between min:max. max is exclusive bound. bounds can be omitted, default to 0 lower bound or maximum upper bound
	mumax3-convert -xrange 50:100 -yrange :100 file.dump
Example: select the bottom layer
	mumax3-convert -zrange :1 file.dump

Output file names are automatically assigned.
*/
package main

import (
	"compress/gzip"
	"flag"
	"github.com/mumax/3/data"
	"github.com/mumax/3/draw"
	"github.com/mumax/3/oommf"
	"github.com/mumax/3/util"
	"log"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
)

var (
	flag_comp      = flag.Int("comp", -1, "Select a component of vector data. 0=x, 1=y, ...")
	flag_show      = flag.Bool("show", false, "Human-readible output to stdout")
	flag_format    = flag.String("f", "%v", "Printf format string")
	flag_png       = flag.Bool("png", false, "PNG output")
	flag_jpeg      = flag.Bool("jpg", false, "JPEG output")
	flag_svg       = flag.Bool("svg", false, "SVG output")
	flag_svgz      = flag.Bool("svgz", false, "SVGZ output (compressed)")
	flag_gnuplot   = flag.Bool("gplot", false, "Gnuplot-compatible output")
	flag_ovf1      = flag.String("ovf", "", `"text" or "binary" OVF1 output`)
	flag_ovf2      = flag.String("ovf2", "", `"text" or "binary" OVF2 output`)
	flag_vtk       = flag.String("vtk", "", `"ascii" or "binary" VTK output`)
	flag_dump      = flag.Bool("dump", false, `output in dump format`)
	flag_csv       = flag.Bool("csv", false, `output in CSV format`)
	flag_json      = flag.Bool("json", false, `output in JSON format`)
	flag_min       = flag.String("min", "auto", `Minimum of color scale: "auto" or value.`)
	flag_max       = flag.String("max", "auto", `Maximum of color scale: "auto" or value.`)
	flag_normalize = flag.Bool("normalize", false, `Normalize vector data to unit length`)
	flag_normpeak  = flag.Bool("normpeak", false, `Scale vector data, maximum to unit length`)
	flag_resize    = flag.String("resize", "", "Resize. E.g.: 4x128x128")
	flag_cropx     = flag.String("xrange", "", "Crop x range min:max (both optional, max=exclusive)")
	flag_cropy     = flag.String("yrange", "", "Crop y range min:max (both optional, max=exclusive)")
	flag_cropz     = flag.String("zrange", "", "Crop z range min:max (both optional, max=exclusive)")
)

var que chan task
var wg sync.WaitGroup

type task struct {
	*data.Slice
	info  data.Meta
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
		var info data.Meta
		var err error

		slice, info, err = oommf.Read(fname)
		if err != nil {
			log.Println(err)
			continue
		}
		wg.Add(1)
		que <- task{slice, info, util.NoExt(fname)}
	}

	// wait for work to finish
	wg.Wait()
}

func work() {
	for task := range que {
		process(task.Slice, task.info, task.fname)
		wg.Done()
	}
}

func open(fname string) *os.File {
	f, err := os.OpenFile(fname, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	util.FatalErr(err)
	return f
}

func process(f *data.Slice, info data.Meta, name string) {
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

	if *flag_svg {
		out := open(name + ".svg")
		defer out.Close()
		draw.SVG(out, f.Vectors())
		haveOutput = true
	}

	if *flag_svgz {
		out1 := open(name + ".svgz")
		defer out1.Close()
		out2 := gzip.NewWriter(out1)
		defer out2.Close()
		draw.SVG(out2, f.Vectors())
		haveOutput = true
	}

	if *flag_gnuplot {
		out := open(name + ".gplot")
		defer out.Close()
		dumpGnuplot(out, f, info)
		haveOutput = true
	}

	if *flag_ovf1 != "" {
		out := open(name + ".ovf")
		defer out.Close()
		oommf.WriteOVF1(out, f, info, *flag_ovf1)
		haveOutput = true
	}

	if *flag_ovf2 != "" {
		out := open(name + ".ovf")
		defer out.Close()
		oommf.WriteOVF2(out, f, *flag_ovf2, info)
		haveOutput = true
	}

	if *flag_vtk != "" {
		out := open(name + ".vts") // vts is the official extension for VTK files containing StructuredGrid data
		defer out.Close()
		dumpVTK(out, f, info, *flag_vtk)
		haveOutput = true
	}

	if *flag_csv {
		out := open(name + ".csv")
		defer out.Close()
		dumpCSV(out, f)
		haveOutput = true
	}

	if *flag_json {
		out := open(name + ".json")
		defer out.Close()
		dumpJSON(out, f)
		haveOutput = true
	}

	if *flag_dump {
		data.MustWriteFile(name+".dump", f, info)
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
		*f = *f.Comp(*flag_comp)
	}
	crop(f)
	if *flag_resize != "" {
		resize(f, *flag_resize)
	}
}

func crop(f *data.Slice) {
	N := f.Size()
	x1, x2 := 0, N[2]
	y1, y2 := 0, N[1]
	z1, z2 := 0, N[0]
	todo := false

	if *flag_cropz != "" {
		z1, z2 = parseRange(*flag_cropz, N[0])
		todo = true
	}
	if *flag_cropy != "" {
		y1, y2 = parseRange(*flag_cropy, N[1])
		todo = true
	}
	if *flag_cropx != "" {
		x1, x2 = parseRange(*flag_cropx, N[2])
		todo = true
	}

	if todo {
		*f = *data.Crop(f, z1, z2, y1, y2, x1, x2)
	}
}

func parseRange(r string, max int) (int, int) {
	a, b := 0, max
	spl := strings.Split(r, ":")
	if len(spl) != 2 {
		log.Fatal("range needs min:max syntax, have:", r)
	}
	if spl[0] != "" {
		a = atoi(spl[0])
	}
	if spl[1] != "" {
		b = atoi(spl[1])
	}
	return a, b
}

func atoi(a string) int {
	i, err := strconv.Atoi(a)
	if err != nil {
		panic(err)
	}
	return i
}

const (
	X = data.X
	Y = data.Y
	Z = data.Z
)
