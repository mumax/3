/*
mumax3-convert converts mumax3 output files to various formats and images.
It also provides basic manipulations like data rescale etc.


Usage

Command-line flags must always preceed the input files:
	mumax3-convert [flags] files
For a overview of flags, run:
	mumax3-convert -help
Example: convert all .ovf files to PNG:
	mumax3-convert -png *.ovf
For scalar data, the color scale is automatically stretched to cover the all values. The values corresponding to minimum and maximum color can be overridden by the -min and -max flags. Values falling outside of this range will be clipped. E.g.:
 	mumax3-convert -png -min=0 -max=1 file.ovf.
The default scalar color map is black,gray,white (minimum value maps to black, maximum to white). This can be overridden by -color. E.g., a rather colorful map:
	mumax3-convert -png -color black,blue,cyan,green,yellow,red,white file.ovf
Example: resize data to a 32 x 32 x 1 mesh, normalize vectors to unit length and convert the result to OOMMF binary output:
	mumax3-convert -resize 32x32x1 -normalize -ovf binary file.ovf
Example: convert all .ovf files to VTK binary saving only the X component. Also output to JPEG in the meanwhile:
	mumax3-convert -comp 0 -vtk binary -jpg *.ovf
Example: convert legacy .dump files to .ovf:
	mumax3-convert -ovf2 *.dump
Example: cut out a piece of the data between min:max. max is exclusive bound. bounds can be omitted, default to 0 lower bound or maximum upper bound
	mumax3-convert -xrange 50:100 -yrange :100 file.ovf
Example: select the bottom layer
	mumax3-convert -zrange :1 file.ovf

Output file names are automatically assigned.
*/
package main

import (
	"compress/gzip"
	"flag"
	"fmt"
	"image/color"
	"io"
	"log"
	"os"
	"path"
	"strconv"
	"strings"

	"github.com/mumax/3/data"
	"github.com/mumax/3/draw"
	"github.com/mumax/3/dump"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/oommf"
	"github.com/mumax/3/util"
)

var (
	flag_comp      = flag.String("comp", "", "Select a component of vector data. (0,1,2 or x,y,z)")
	flag_show      = flag.Bool("show", false, "Human-readible output to stdout")
	flag_format    = flag.String("f", "%v", "Printf format string")
	flag_png       = flag.Bool("png", false, "PNG output")
	flag_jpeg      = flag.Bool("jpg", false, "JPEG output")
	flag_gif       = flag.Bool("gif", false, "GIF output")
	flag_svg       = flag.Bool("svg", false, "SVG output")
	flag_svgz      = flag.Bool("svgz", false, "SVGZ output (compressed)")
	flag_gnuplot   = flag.Bool("gplot", false, "Gnuplot-compatible output")
	flag_ovf1      = flag.String("ovf", "", `"text" or "binary" OVF1 output`)
	flag_omf       = flag.String("omf", "", `"text" or "binary" OVF1 output`)
	flag_ovf2      = flag.String("ovf2", "", `"text" or "binary" OVF2 output`)
	flag_vtk       = flag.String("vtk", "", `"ascii" or "binary" VTK output`)
	flag_dump      = flag.Bool("dump", false, `output in dump format`)
	flag_csv       = flag.Bool("csv", false, `output in CSV format`)
	flag_json      = flag.Bool("json", false, `output in JSON format`)
	flag_min       = flag.String("min", "auto", `Minimum of color scale: "auto" or value.`)
	flag_max       = flag.String("max", "auto", `Maximum of color scale: "auto" or value.`)
	flag_normalize = flag.Bool("normalize", false, `Normalize vector data to unit length`)
	flag_normpeak  = flag.Bool("normpeak", false, `Scale vector data, maximum to unit length`)
	flag_resize    = flag.String("resize", "", "Resize. E.g.: 128x128x4")
	flag_cropx     = flag.String("xrange", "", "Crop x range min:max (both optional, max=exclusive)")
	flag_cropy     = flag.String("yrange", "", "Crop y range min:max (both optional, max=exclusive)")
	flag_cropz     = flag.String("zrange", "", "Crop z range min:max (both optional, max=exclusive)")
	flag_dir       = flag.String("o", "", "Save all output in this directory")
	flag_arrows    = flag.Int("arrows", 0, "Arrow size for vector bitmap image output")
	flag_color     = flag.String("color", "black,gray,white", "Colormap for scalar image output.")
)

var (
	colormap []color.RGBA
)

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

	colormap = parseColors(*flag_color)

	// politely try to make the output directory
	if *flag_dir != "" {
		_ = os.Mkdir(*flag_dir, 0777)
	}

	// determine which outputs we want
	var wantOut []output
	for flag, out := range outputs {
		if *flag {
			wantOut = append(wantOut, out)
		}
	}
	if len(wantOut) == 0 && *flag_show == false {
		log.Fatal("no output format specified (e.g.: -png)")
	}

	// read all input files and put them in the task que
	for _, fname := range flag.Args() {
		for _, outp := range wantOut {
			fname := fname // closure caveats
			Queue(func() {
				doFile(fname, outp)
			})
		}
	}

	// wait for work to finish
	Wait()
}

func doFile(infname string, outp output) {
	// determine output file
	outfname := util.NoExt(infname) + outp.Ext
	if *flag_dir != "" {
		outfname = *flag_dir + "/" + path.Base(outfname)
	}

	msg := infname + "\t-> " + outfname
	defer func() { log.Println(msg) }()

	var slice *data.Slice
	var info data.Meta
	var err error

	switch path.Ext(infname) {
	default:
		msg = "[fail] " + msg + ": skipping unsupported type: " + path.Ext(infname)
		return
	case ".ovf", ".omf", ".ovf2":
		slice, info, err = oommf.ReadFile(infname)
	case ".dump":
		slice, info, err = dump.ReadFile(infname)
	}

	if err != nil {
		msg += ": " + err.Error()
		return
	}

	out, err := httpfs.Create(outfname)
	if err != nil {
		msg += "[fail] " + msg + ": " + err.Error()
		return
	}
	defer out.Close()

	outp.Convert(slice, info, out)
	msg = "[ ok ] " + msg

}

type output struct {
	Ext     string
	Convert func(*data.Slice, data.Meta, io.Writer)
}

var outputs = map[*bool]output{
	flag_png: {".png", renderPNG},
	//flag_jpeg      :
	//flag_gif       :
	//flag_svg       :
	//flag_svgz      :
	//flag_gnuplot   :
	//flag_ovf1      :
	//flag_omf       :
	//flag_ovf2      :
	//flag_vtk       :
	//flag_dump      :
	//flag_csv       :
	//flag_json      :
}

//func work() {
//	for task := range que {
//		process(task.Slice, task.info, task.fname)
//		wg.Done()
//	}
//}

func open(fname string) *os.File {
	f, err := os.OpenFile(fname, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	util.FatalErr(err)
	return f
}

func renderPNG(f *data.Slice, info data.Meta, out io.Writer) {
	draw.RenderFormat(out, f, *flag_min, *flag_max, *flag_arrows, ".png", colormap...)
}

func process(f *data.Slice, info data.Meta, name string) {
	preprocess(f)

	haveOutput := false

	if *flag_jpeg {
		draw.RenderFile(name+".jpg", f, *flag_min, *flag_max, *flag_arrows, colormap...)
		haveOutput = true
	}

	if *flag_png {
		haveOutput = true
	}

	if *flag_gif {
		draw.RenderFile(name+".gif", f, *flag_min, *flag_max, *flag_arrows, colormap...)
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

	if *flag_omf != "" {
		out := open(name + ".omf")
		defer out.Close()
		oommf.WriteOVF1(out, f, info, *flag_omf)
		haveOutput = true
	}

	if *flag_ovf2 != "" {
		out := open(name + ".ovf")
		defer out.Close()
		oommf.WriteOVF2(out, f, info, *flag_ovf2)
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
		dump.MustWriteFile(name+".dump", f, info)
		haveOutput = true
	}

	if !haveOutput || *flag_show {
		fmt.Println(info)
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
	if *flag_comp != "" {
		*f = *f.Comp(parseComp(*flag_comp))
	}
	crop(f)
	if *flag_resize != "" {
		resize(f, *flag_resize)
	}
}

func parseComp(c string) int {
	if i, err := strconv.Atoi(c); err == nil {
		return i
	}
	switch c {
	default:
		log.Fatal("illegal component:", c, "(need x, y or z)")
		panic(0)
	case "x", "X":
		return 0
	case "y", "Y":
		return 1
	case "z", "Z":
		return 2
	}
}

func crop(f *data.Slice) {
	N := f.Size()
	// default ranges
	x1, x2 := 0, N[X]
	y1, y2 := 0, N[Y]
	z1, z2 := 0, N[Z]
	havework := false

	if *flag_cropz != "" {
		z1, z2 = parseRange(*flag_cropz, N[Z])
		havework = true
	}
	if *flag_cropy != "" {
		y1, y2 = parseRange(*flag_cropy, N[Y])
		havework = true
	}
	if *flag_cropx != "" {
		x1, x2 = parseRange(*flag_cropx, N[X])
		havework = true
	}

	if havework {
		*f = *data.Crop(f, x1, x2, y1, y2, z1, z2)
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

func parseColors(s string) (m []color.RGBA) {
	words := strings.Split(s, ",")
	for _, w := range words {
		m = append(m, parseColor(w))
	}
	return
}

func parseColor(s string) color.RGBA {
	if c, ok := colors[s]; ok {
		return c
	}
	fmt.Println("refusing to use ugly color '" + s + "', options are:")
	for k, _ := range colors {
		fmt.Println(k)
	}
	log.Fatal("illegal color")
	return color.RGBA{}
}

var colors = map[string]color.RGBA{
	"white":       color.RGBA{R: 255, G: 255, B: 255, A: 255},
	"black":       color.RGBA{R: 0, G: 0, B: 0, A: 255},
	"transparent": color.RGBA{R: 0, G: 0, B: 0, A: 0},
	"red":         color.RGBA{R: 255, G: 0, B: 0, A: 255},
	"green":       color.RGBA{R: 0, G: 255, B: 0, A: 255},
	"blue":        color.RGBA{R: 0, G: 0, B: 255, A: 255},
	"lightred":    color.RGBA{R: 255, G: 127, B: 127, A: 255},
	"lightgreen":  color.RGBA{R: 127, G: 255, B: 127, A: 255},
	"lightblue":   color.RGBA{R: 127, G: 127, B: 255, A: 255},
	"yellow":      color.RGBA{R: 255, G: 255, B: 0, A: 255},
	"darkyellow":  color.RGBA{R: 127, G: 127, B: 0, A: 255},
	"cyan":        color.RGBA{R: 0, G: 255, B: 255, A: 255},
	"darkcyan":    color.RGBA{R: 0, G: 127, B: 127, A: 255},
	"magenta":     color.RGBA{R: 255, G: 0, B: 255, A: 255},
	"darkmagenta": color.RGBA{R: 127, G: 0, B: 127, A: 255},
	"gray":        color.RGBA{R: 127, G: 127, B: 127, A: 255},
}
