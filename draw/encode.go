package draw

import (
	"bufio"
	"fmt"
	"github.com/mumax/3/data"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"io"
	"os"
	"path"
	"strings"
)

func RenderFile(fname string, f *data.Slice, min, max string, arrowSize int, colormap ...color.RGBA) error {
	out, err := os.Create(fname)
	if err != nil {
		return err
	}
	defer out.Close()
	return RenderFormat(out, f, min, max, arrowSize, fname, colormap...)
}

func RenderFormat(out io.Writer, f *data.Slice, min, max string, arrowSize int, format string, colormap ...color.RGBA) error {
	var codecs = map[string]codec{".png": PNG, ".jpg": JPEG100, ".gif": GIF256}
	ext := strings.ToLower(path.Ext(format))
	enc := codecs[ext]
	if enc == nil {
		return fmt.Errorf("render: unhandled image type: " + ext)
	}
	return Render(out, f, min, max, arrowSize, enc, colormap...)
}

// encodes an image
type codec func(io.Writer, image.Image) error

// Render data and encode with arbitrary codec.
func Render(out io.Writer, f *data.Slice, min, max string, arrowSize int, encode codec, colormap ...color.RGBA) error {
	img := Image(f, min, max, arrowSize, colormap...)
	buf := bufio.NewWriter(out)
	defer buf.Flush()
	return encode(buf, img)
}

// full-quality jpeg codec, passable to Render()
func JPEG100(w io.Writer, img image.Image) error {
	return jpeg.Encode(w, img, &jpeg.Options{100})
}

// full quality gif coded, passable to Render()
func GIF256(w io.Writer, img image.Image) error {
	return gif.Encode(w, img, &gif.Options{256, nil, nil})
}

// png codec, passable to Render()
func PNG(w io.Writer, img image.Image) error {
	return png.Encode(w, img)
}
