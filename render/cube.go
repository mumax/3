package render

import (
	gl "github.com/chsc/gogl/gl21"
)

type Cube struct {
	BoundingBox
}

// Issue to OpenGL.
func (c *Cube) Render() {
	cx, cy, cz := c.Center.X, c.Center.Y, c.Center.Z
	rx, ry, rz := c.Radius.X, c.Radius.Y, c.Radius.Z

	gl.Begin(gl.QUADS)
	{
		Normal3f(0, 0, 1)
		gl.TexCoord2f(0, 0)
		Vertex3f(cx+rx, cy+ry, cz+rz)
		gl.TexCoord2f(0, 1)
		Vertex3f(cx-rx, cy+ry, cz+rz)
		gl.TexCoord2f(1, 1)
		Vertex3f(cx-rx, cy-ry, cz+rz)
		gl.TexCoord2f(1, 0)
		Vertex3f(cx+rx, cy-ry, cz+rz)

		Normal3f(0, 0, -1)
		gl.TexCoord2f(0, 0)
		Vertex3f(cx+rx, cy+ry, cz-rz)
		gl.TexCoord2f(0, 1)
		Vertex3f(cx+rx, cy-ry, cz-rz)
		gl.TexCoord2f(1, 1)
		Vertex3f(cx-rx, cy-ry, cz-rz)
		gl.TexCoord2f(1, 0)
		Vertex3f(cx-rx, cy+ry, cz-rz)

		Normal3f(0, 1, 0)
		gl.TexCoord2f(0, 0)
		Vertex3f(cx+rx, cy+ry, cz+rz)
		gl.TexCoord2f(0, 1)
		Vertex3f(cx+rx, cy+ry, cz-rz)
		gl.TexCoord2f(1, 1)
		Vertex3f(cx-rx, cy+ry, cz-rz)
		gl.TexCoord2f(1, 0)
		Vertex3f(cx-rx, cy+ry, cz+rz)

		Normal3f(0, -1, 0)
		gl.TexCoord2f(0, 0)
		Vertex3f(cx-rx, cy-ry, cz+rz)
		gl.TexCoord2f(0, 1)
		Vertex3f(cx-rx, cy-ry, cz-rz)
		gl.TexCoord2f(1, 1)
		Vertex3f(cx+rx, cy-ry, cz-rz)
		gl.TexCoord2f(1, 0)
		Vertex3f(cx+rx, cy-ry, cz+rz)

		Normal3f(1, 0, 0)
		gl.TexCoord2f(0, 0)
		Vertex3f(cx+rx, cy-ry, cz+rz)
		gl.TexCoord2f(0, 1)
		Vertex3f(cx+rx, cy-ry, cz-rz)
		gl.TexCoord2f(1, 1)
		Vertex3f(cx+rx, cy+ry, cz-rz)
		gl.TexCoord2f(1, 0)
		Vertex3f(cx+rx, cy+ry, cz+rz)

		Normal3f(-1, 0, 0)
		gl.TexCoord2f(0, 0)
		Vertex3f(cx-rx, cy+ry, cz+rz)
		gl.TexCoord2f(0, 1)
		Vertex3f(cx-rx, cy+ry, cz-rz)
		gl.TexCoord2f(1, 1)
		Vertex3f(cx-rx, cy-ry, cz-rz)
		gl.TexCoord2f(1, 0)
		Vertex3f(cx-rx, cy-ry, cz+rz)
	}
	gl.End()
}
