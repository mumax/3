package render

import (
	gl "github.com/chsc/gogl/gl21"
)

func Cube(cx, cy, cz, rx, ry, rz float32) {
	gl.Begin(gl.POLYGON)
	{
		Normal3f(0, 0, 1)
		Vertex3f(cx+rx, cy+ry, cz+rz)
		Vertex3f(cx-rx, cy+ry, cz+rz)
		Vertex3f(cx-rx, cy-ry, cz+rz)
		Vertex3f(cx+rx, cy-ry, cz+rz)
	}
	gl.End()

	gl.Begin(gl.POLYGON)
	{
		Normal3f(0, 0, -1)
		Vertex3f(cx+rx, cy+ry, cz-rz)
		Vertex3f(cx+rx, cy-ry, cz-rz)
		Vertex3f(cx-rx, cy-ry, cz-rz)
		Vertex3f(cx-rx, cy+ry, cz-rz)
	}
	gl.End()

	gl.Begin(gl.POLYGON)
	{
		Normal3f(0, 1, 0)
		Vertex3f(cx+rx, cy+ry, cz+rz)
		Vertex3f(cx+rx, cy+ry, cz-rz)
		Vertex3f(cx-rx, cy+ry, cz-rz)
		Vertex3f(cx-rx, cy+ry, cz+rz)
	}
	gl.End()

	gl.Begin(gl.POLYGON)
	{
		Normal3f(0, -1, 0)
		Vertex3f(cx-rx, cy-ry, cz+rz)
		Vertex3f(cx-rx, cy-ry, cz-rz)
		Vertex3f(cx+rx, cy-ry, cz-rz)
		Vertex3f(cx+rx, cy-ry, cz+rz)
	}
	gl.End()

	gl.Begin(gl.POLYGON)
	{
		Normal3f(1, 0, 0)
		Vertex3f(cx+rx, cy-ry, cz+rz)
		Vertex3f(cx+rx, cy-ry, cz-rz)
		Vertex3f(cx+rx, cy+ry, cz-rz)
		Vertex3f(cx+rx, cy+ry, cz+rz)
	}
	gl.End()

	gl.Begin(gl.POLYGON)
	{
		Normal3f(-1, 0, 0)
		Vertex3f(cx-rx, cy+ry, cz+rz)
		Vertex3f(cx-rx, cy+ry, cz-rz)
		Vertex3f(cx-rx, cy-ry, cz-rz)
		Vertex3f(cx-rx, cy-ry, cz+rz)
	}
	gl.End()
}
