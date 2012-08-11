package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"nimble-cube/core"
	"nimble-cube/dump"
	"os"
	"path"
)

func dumpImage(f *dump.Frame, file string) {
	var img *image.NRGBA
	{
		dim := f.Size[0]
		switch dim {
		default:
			core.Fatal(fmt.Errorf("unsupported number of components: %v", dim))
		case 3:
			img = DrawVectors(f.Vectors())
		case 1:
			min, max := extrema(f.Data)
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
