package render

import gl "github.com/chsc/gogl/gl21"

func DrawTestScene() {

	ambient := []gl.Float{0.7, 0.7, 0.7, 1}
	diffuse := []gl.Float{1, 1, 1, 1}
	lightpos := []gl.Float{0.2, 0.5, 1, 1}
	gl.Lightfv(gl.LIGHT0, gl.AMBIENT, &ambient[0])
	gl.Lightfv(gl.LIGHT0, gl.DIFFUSE, &diffuse[0])
	gl.Lightfv(gl.LIGHT0, gl.POSITION, &lightpos[0])
	gl.Enable(gl.LIGHT0)

	ambdiff := []gl.Float{0.5, 0.5, 0.0, 1}
	gl.Materialfv(gl.FRONT_AND_BACK, gl.AMBIENT_AND_DIFFUSE, &ambdiff[0])

	const r = 0.5
	gl.ClearColor(1, 1, 1, 1)
	gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
	Color3f(1, 0, 0)
	Cube(0, 0, 0, r, r, r)
}
