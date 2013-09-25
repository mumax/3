package main

// Image output.
// Author: Arne Vansteenkiste

import (
	"bufio"
	"github.com/mumax/3/data"
	"github.com/mumax/3/draw"
	"image/jpeg"
	"image/png"
	"io"
)

func dumpPNG(out io.Writer, f *data.Slice) error {
	img := draw.Image(f, *flag_min, *flag_max)
	buf := bufio.NewWriter(out)
	defer buf.Flush()
	return png.Encode(buf, img)
}

func dumpJPEG(out io.Writer, f *data.Slice) error {
	img := draw.Image(f, *flag_min, *flag_max)
	buf := bufio.NewWriter(out)
	defer buf.Flush()
	return jpeg.Encode(buf, img, &jpeg.Options{Quality: 100})
}
