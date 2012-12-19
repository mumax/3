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
		for _, v := range p.vertex {
			Vertex3f(v[0], v[1], v[2])
		}
	}
}

func X1Face(x, y, z, rx, ry, rz float32, col color.NRGBA) Poly {
	return Poly{[3]float32{},
		[4][3]float32{
			{z - rz, y - ry, x - rx},
			{z - rz, y + ry, x - rx},
			{z + rz, y + ry, x - rx},
			{z + rz, y - ry, x - rx}}, col}
}

func X2Face(x, y, z, rx, ry, rz float32, col color.NRGBA) Poly {
	return Poly{[3]float32{},
		[4][3]float32{
			{z - rz, y - ry, x + rx},
			{z + rz, y - ry, x + rx},
			{z + rz, y + ry, x + rx},
			{z - rz, y + ry, x + rx}}, col}
}

func Y1Face(x, y, z, rx, ry, rz float32, col color.NRGBA) Poly {
	return Poly{[3]float32{},
		[4][3]float32{
			{z - rz, y - ry, x - rx},
			{z + rz, y - ry, x - rx},
			{z + rz, y - ry, x + rx},
			{z - rz, y - ry, x + rx}}, col}
}

func Y2Face(x, y, z, rx, ry, rz float32, col color.NRGBA) Poly {
	return Poly{[3]float32{},
		[4][3]float32{
			{z - rz, y + ry, x - rx},
			{z - rz, y + ry, x + rx},
			{z + rz, y + ry, x + rx},
			{z + rz, y + ry, x - rx}}, col}
}

func Z1Face(x, y, z, rx, ry, rz float32, col color.NRGBA) Poly {
	return Poly{[3]float32{},
		[4][3]float32{
			{z - rz, y - ry, x - rx},
			{z - rz, y - ry, x + rx},
			{z - rz, y + ry, x + rx},
			{z - rz, y + ry, x - rx}}, col}
}

func Z2Face(x, y, z, rx, ry, rz float32, col color.NRGBA) Poly {
	return Poly{[3]float32{},
		[4][3]float32{
			{z + rz, y - ry, x - rx},
			{z + rz, y + ry, x - rx},
			{z + rz, y + ry, x + rx},
			{z + rz, y - ry, x + rx}}, col}
}
