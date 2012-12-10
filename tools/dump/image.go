package main

// Image output.
// Author: Arne Vansteenkiste

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/dump"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"path"
	"strconv"
)

func dumpImage(f *dump.Frame, file string) {
	var img *image.NRGBA
	{
		dim := f.NComp()
		switch dim {
		default:
			core.Fatal(fmt.Errorf("unsupported number of components: %v", dim))
		case 3:
			img = DrawVectors(f.Vectors())
		case 1:
			min, max := extrema(f.Data)
			if *flag_min != "auto" {
				m, err := strconv.ParseFloat(*flag_min, 32)
				core.Fatal(err)
				min = float32(m)
			}
			if *flag_max != "auto" {
				m, err := strconv.ParseFloat(*flag_max, 32)
				core.Fatal(err)
				max = float32(m)
			}
			img = DrawFloats(f.Floats(), min, max)
		}
	}

	out, err := os.OpenFile(file, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	core.Fatal(err)
	defer out.Close()

	ext := path.Ext(file)
	switch ext {
	default:
		core.Fatal(fmt.Errorf("unsupported image type: %v", ext))
	case ".png":
		core.Fatal(png.Encode(out, img))
	case ".jpg":
		core.Fatal(jpeg.Encode(out, img, nil))
	}
}
