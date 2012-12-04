package render

import gl "github.com/chsc/gogl/gl21"

// Wraps gl.Vertex3f
func Vertex3f(x, y, z float32) {
	gl.Vertex3f(gl.Float(x), gl.Float(y), gl.Float(z))
}

// Wraps gl.Normal3f
func Normal3f(x, y, z float32) {
	gl.Normal3f(gl.Float(x), gl.Float(y), gl.Float(z))
}

func Color3f(r, g, b float32) {
	gl.Color3f(gl.Float(r), gl.Float(g), gl.Float(b))
}
