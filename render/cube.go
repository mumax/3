package render

import (
	gl "github.com/chsc/gogl/gl21"
)

// Renders a cube with given center and radius.
func Cube(cx, cy, cz, rx, ry, rz float32) {
	gl.Begin(gl.QUADS)
	{
		Normal3f(0, 0, 1)
		Vertex3f(cx+rx, cy+ry, cz+rz)
		Vertex3f(cx-rx, cy+ry, cz+rz)
		Vertex3f(cx-rx, cy-ry, cz+rz)
		Vertex3f(cx+rx, cy-ry, cz+rz)

		Normal3f(0, 0, -1)
		Vertex3f(cx+rx, cy+ry, cz-rz)
		Vertex3f(cx+rx, cy-ry, cz-rz)
		Vertex3f(cx-rx, cy-ry, cz-rz)
		Vertex3f(cx-rx, cy+ry, cz-rz)

		Normal3f(0, 1, 0)
		Vertex3f(cx+rx, cy+ry, cz+rz)
		Vertex3f(cx+rx, cy+ry, cz-rz)
		Vertex3f(cx-rx, cy+ry, cz-rz)
		Vertex3f(cx-rx, cy+ry, cz+rz)

		Normal3f(0, -1, 0)
		Vertex3f(cx-rx, cy-ry, cz+rz)
		Vertex3f(cx-rx, cy-ry, cz-rz)
		Vertex3f(cx+rx, cy-ry, cz-rz)
		Vertex3f(cx+rx, cy-ry, cz+rz)

		Normal3f(1, 0, 0)
		Vertex3f(cx+rx, cy-ry, cz+rz)
		Vertex3f(cx+rx, cy-ry, cz-rz)
		Vertex3f(cx+rx, cy+ry, cz-rz)
		Vertex3f(cx+rx, cy+ry, cz+rz)

		Normal3f(-1, 0, 0)
		Vertex3f(cx-rx, cy+ry, cz+rz)
		Vertex3f(cx-rx, cy+ry, cz-rz)
		Vertex3f(cx-rx, cy-ry, cz-rz)
		Vertex3f(cx-rx, cy-ry, cz+rz)
	}
	gl.End()
}

// Wraps gl.Vertex3f
func Vertex3f(x, y, z float32) {
	gl.Vertex3f(gl.Float(x), gl.Float(y), gl.Float(z))
}

// Wraps gl.Normal3f
func Normal3f(x, y, z float32) {
	gl.Normal3f(gl.Float(x), gl.Float(y), gl.Float(z))
}

// Wraps gl.Color3f
func Color3f(r, g, b float32) {
	gl.Color3f(gl.Float(r), gl.Float(g), gl.Float(b))
}
