package main

import (
	"flag"
	"io"
	"nimble-cube/core"
	"nimble-cube/dump"
	"os"
	"path"
)

var (
	flag_crc     = flag.Bool("crc", true, "Generate/check CRC checksums")
	flag_show    = flag.Bool("show", false, "Human-readible output to stdout")
	flag_format  = flag.String("f", "%v", "Printf format string")
	flag_png     = flag.Bool("png", false, "PNG output")
	flag_jpeg    = flag.Bool("jpeg", false, "JPEG output")
	flag_gnuplot = flag.Bool("gplot", false, "Gnuplot-compatible output")
	flag_min     = flag.String("min", "auto", `Minimum of color scale: "auto" or value.`)
	flag_max     = flag.String("max", "auto", `Maximum of color scale: "auto" or value.`)
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
	for err != io.EOF {
		core.Fatal(err)
		process(&r.Frame, name)
		err = r.Read()
	}
}

func process(f *dump.Frame, name string) {
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

	if !haveOutput || *flag_show {
		f.Fprintf(os.Stdout, *flag_format)
		haveOutput = true
	}
}

func noExt(file string) string {
	ext := path.Ext(file)
	return file[:len(file)-len(ext)]
}
