package render

import (
	gl "github.com/chsc/gogl/gl21"
)

// A point in 3D space.
type Vertex struct {
	X, Y, Z float32
}

// Issue the vertex to OpenGL.
func (v *Vertex) Render() {
	gl.Vertex3f(gl.Float(v.X), gl.Float(v.Y), gl.Float(v.Z))
}

// Multiply-add: a+=s*b.
func (a *Vertex) MAdd(s float32, b Vertex) {
	a.X += s * b.X
	a.Y += s * b.Y
	a.Z += s * b.Z
}
