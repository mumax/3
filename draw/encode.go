package draw

import (
	"bufio"
	"errors"
	"github.com/mumax/3/data"
	"image"
	"image/gif"
	"image/jpeg"
	"image/png"
	"io"
	"os"
	"path"
	"strings"
)

func RenderFile(fname string, f *data.Slice, min, max string, arrowSize int) error {
	codecs := map[string]codec{".png": PNG, ".jpg": JPEG100, ".gif": GIF256}
	ext := strings.ToLower(path.Ext(fname))
	enc := codecs[ext]
	if enc == nil {
		return errors.New("renderfile: unhandled image type: " + ext)
	}
	out, err := os.OpenFile(fname, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		return err
	}
	defer out.Close()
	return Render(out, f, min, max, arrowSize, enc)
}

// encodes an image
type codec func(io.Writer, image.Image) error

// Render data and encode with arbitrary codec.
func Render(out io.Writer, f *data.Slice, min, max string, arrowSize int, encode codec) error {
	img := Image(f, min, max, arrowSize)
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
