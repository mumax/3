package render

import (
	"code.google.com/p/mx3/core"
	gl "github.com/chsc/gogl/gl21"
	"image"
	"image/png"
	"log"
	"os"
)

func Screenshot() {
	log.Println("screenshot")
	img := image.NewNRGBA(image.Rect(0, 0, Width, Height))
	gl.ReadPixels(0, 0, gl.Sizei(Width), gl.Sizei(Height), gl.RGBA, gl.UNSIGNED_INT_8_8_8_8, gl.Pointer(&img.Pix[0]))
	//go func(){
	// reverse byte order, opengl seems to use little endian.
	pix := img.Pix
	for i := 0; i < len(pix); i += 4 {
		pix[i+0], pix[i+1], pix[i+2], pix[i+3] = pix[i+3], pix[i+2], pix[i+1], pix[i+0]
	}
	f, err := os.OpenFile("scrot.png", os.O_WRONLY|os.O_TRUNC|os.O_CREATE, 0666)
	core.Fatal(err)
	defer f.Close()
	core.Fatal(png.Encode(f, img))
	//}()
}
