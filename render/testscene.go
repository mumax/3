package render

import (
	gl "github.com/chsc/gogl/gl21"
)

func DrawTestScene() {

	ambient := []gl.Float{0.7, 0.7, 0.7, 1}
	diffuse := []gl.Float{1, 1, 1, 1}
	lightpos := []gl.Float{0.2, 0.5, 1, 1}
	gl.Lightfv(gl.LIGHT0, gl.AMBIENT, &ambient[0])
	gl.Lightfv(gl.LIGHT0, gl.DIFFUSE, &diffuse[0])
	gl.Lightfv(gl.LIGHT0, gl.POSITION, &lightpos[0])
	gl.Enable(gl.LIGHT0)
	ambdiff := []gl.Float{0.0, 0.5, 0.5, 1}
	gl.Materialfv(gl.FRONT_AND_BACK, gl.AMBIENT_AND_DIFFUSE, &ambdiff[0])

	gl.ClearColor(0, 0, 0, 0)
	gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

	(&Cube{BoundingBox{Vertex{0, 0, -1}, Vertex{10, 10, 0.2}}}).Render()

	(&Cube{BoundingBox{Vertex{5, 0, 0}, Vertex{0.1, 0.1, 1}}}).Render()

	(&Cube{BoundingBox{Vertex{-5, 0, 0}, Vertex{0.1, 0.1, 0.1}}}).Render()

	(&Cube{BoundingBox{Vertex{1, 6, 0}, Vertex{0.1, 0.1, 1}}}).Render()
	(&Cube{BoundingBox{Vertex{0, 5, 0}, Vertex{0.1, 0.1, 1}}}).Render()

	(&Cube{BoundingBox{Vertex{1, -6, 0}, Vertex{0.1, 0.1, 0.1}}}).Render()
	(&Cube{BoundingBox{Vertex{0, -5, 0}, Vertex{0.1, 0.1, 0.1}}}).Render()

	(&Cube{BoundingBox{Vertex{0, 0, 10}, Vertex{0.2, 0.2, 0.2}}}).Render()

}
