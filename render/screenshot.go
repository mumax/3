package render

import (
	"code.google.com/p/mx3/core"
	"fmt"
	gl "github.com/chsc/gogl/gl21"
	"image"
	"image/png"
	"log"
	"os"
)

var scroti int // count screenshots

func Screenshot() {
	log.Println("screenshot")
	img := image.NewNRGBA(image.Rect(0, 0, Width, Height))
	gl.ReadPixels(0, 0, gl.Sizei(Width), gl.Sizei(Height), gl.RGBA, gl.UNSIGNED_INT_8_8_8_8, gl.Pointer(&img.Pix[0]))
	// reverse byte order, opengl seems to use little endian.
	pix := img.Pix
	for i := 0; i < len(pix); i += 4 {
		pix[i+0], pix[i+1], pix[i+2], pix[i+3] = pix[i+3], pix[i+2], pix[i+1], pix[i+0]
	}
	fname := fmt.Sprintf("frame%04d.png", scroti)
	scroti++
	f, err := os.OpenFile(fname, os.O_WRONLY|os.O_TRUNC|os.O_CREATE, 0666)
	core.Fatal(err)
	defer f.Close()
	core.Fatal(png.Encode(f, img))
}
