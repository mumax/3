package render

import (
	gl "github.com/chsc/gogl/gl21"
	"image/color"
)

func DrawTestScene() {

	ambient := []gl.Float{0.7, 0.7, 0.7, 1}
	diffuse := []gl.Float{1, 1, 1, 1}
	lightpos := []gl.Float{0.2, 0.5, 1, 1}
	gl.Lightfv(gl.LIGHT0, gl.AMBIENT, &ambient[0])
	gl.Lightfv(gl.LIGHT0, gl.DIFFUSE, &diffuse[0])
	gl.Lightfv(gl.LIGHT0, gl.POSITION, &lightpos[0])
	gl.Enable(gl.LIGHT0)

	gl.ClearColor(1, 1, 1, 1)
	gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

	const r = 0.5
	const z = r
	red := color.NRGBA{R: 255, A: 255}
	blue := color.NRGBA{B: 255, A: 255}

	gl.Begin(gl.QUADS)
	(&Poly{[4][3]float32{{-r, -r, z}, {r, -r, z}, {r, r, z}, {-r, r, z}}, red}).Render()
	(&Poly{[4][3]float32{{-r, -r, -z}, {-r, r, -z}, {r, r, -z}, {r, -r, -z}}, blue}).Render()
	gl.End()

}
