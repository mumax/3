package render

import (
	gl "github.com/chsc/gogl/gl21"
	"image/color"
)

type Poly struct {
	normal [3]float32
	vertex [4][3]float32
	color.NRGBA
}

var colbuf = [4]float32{0, 0, 0, 1}

func (p *Poly) Render() {
	if p.A != 0 {
		colbuf[0], colbuf[1], colbuf[2] = float32(p.R)/255, float32(p.G)/255, float32(p.B)/255
		gl.Materialfv(gl.FRONT_AND_BACK, gl.AMBIENT_AND_DIFFUSE, (*gl.Float)(&(colbuf[0])))
		Normal3f(p.normal[0], p.normal[1], p.normal[2])
		for _, v := range p.vertex {
			Vertex3f(v[0], v[1], v[2])
		}
	}
}

func X1Face(x, y, z, rx, ry, rz float32, col color.NRGBA) Poly {
	return Poly{[3]float32{0, 0, -1},
		[4][3]float32{
			{z - rz, y - ry, x - rx},
			{z - rz, y + ry, x - rx},
			{z + rz, y + ry, x - rx},
			{z + rz, y - ry, x - rx}}, col}
}

func X2Face(x, y, z, rx, ry, rz float32, col color.NRGBA) Poly {
	return Poly{[3]float32{0, 0, 1},
		[4][3]float32{
			{z - rz, y - ry, x + rx},
			{z + rz, y - ry, x + rx},
			{z + rz, y + ry, x + rx},
			{z - rz, y + ry, x + rx}}, col}
}

func Y1Face(x, y, z, rx, ry, rz float32, col color.NRGBA) Poly {
	return Poly{[3]float32{0, -1, 0},
		[4][3]float32{
			{z - rz, y - ry, x - rx},
			{z + rz, y - ry, x - rx},
			{z + rz, y - ry, x + rx},
			{z - rz, y - ry, x + rx}}, col}
}

func Y2Face(x, y, z, rx, ry, rz float32, col color.NRGBA) Poly {
	return Poly{[3]float32{0, 1, 0},
		[4][3]float32{
			{z - rz, y + ry, x - rx},
			{z - rz, y + ry, x + rx},
			{z + rz, y + ry, x + rx},
			{z + rz, y + ry, x - rx}}, col}
}

func Z1Face(x, y, z, rx, ry, rz float32, col color.NRGBA) Poly {
	return Poly{[3]float32{-1, 0, 0},
		[4][3]float32{
			{z - rz, y - ry, x - rx},
			{z - rz, y - ry, x + rx},
			{z - rz, y + ry, x + rx},
			{z - rz, y + ry, x - rx}}, col}
}

func Z2Face(x, y, z, rx, ry, rz float32, col color.NRGBA) Poly {
	return Poly{[3]float32{1, 0, 0},
		[4][3]float32{
			{z + rz, y - ry, x - rx},
			{z + rz, y + ry, x - rx},
			{z + rz, y + ry, x + rx},
			{z + rz, y - ry, x + rx}}, col}
}

// Wraps gl.Vertex3f
func Vertex3f(x, y, z float32) {
	gl.Vertex3f(gl.Float(x), gl.Float(y), gl.Float(z))
}

// Wraps gl.Normal3f
func Normal3f(x, y, z float32) {
	gl.Normal3f(gl.Float(x), gl.Float(y), gl.Float(z))
}
