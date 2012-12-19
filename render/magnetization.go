package render

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/dump"
	gl "github.com/chsc/gogl/gl21"
	"image/color"
	"os"
)

type Poly struct {
	vertex [4][3]float32
	color.NRGBA
}

var (
	polys []Poly
	Frame *dump.Frame
)

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
	return Poly{[4][3]float32{
		{z - rz, y - ry, x - rx},
		{z - rz, y + ry, x - rx},
		{z + rz, y + ry, x - rx},
		{z + rz, y - ry, x - rx}}, col}
}

func X2Face(x, y, z, rx, ry, rz float32, col color.NRGBA) Poly {
	return Poly{[4][3]float32{
		{z - rz, y - ry, x + rx},
		{z + rz, y - ry, x + rx},
		{z + rz, y + ry, x + rx},
		{z - rz, y + ry, x + rx}}, col}
}

func Y1Face(x, y, z, rx, ry, rz float32, col color.NRGBA) Poly {
	return Poly{[4][3]float32{
		{z - rz, y - ry, x - rx},
		{z + rz, y - ry, x - rx},
		{z + rz, y - ry, x + rx},
		{z - rz, y - ry, x + rx}}, col}
}

func Y2Face(x, y, z, rx, ry, rz float32, col color.NRGBA) Poly {
	return Poly{[4][3]float32{
		{z - rz, y + ry, x - rx},
		{z - rz, y + ry, x + rx},
		{z + rz, y + ry, x + rx},
		{z + rz, y + ry, x - rx}}, col}
}

func Z1Face(x, y, z, rx, ry, rz float32, col color.NRGBA) Poly {
	return Poly{[4][3]float32{
		{z - rz, y - ry, x - rx},
		{z - rz, y - ry, x + rx},
		{z - rz, y + ry, x + rx},
		{z - rz, y + ry, x - rx}}, col}
}

func Z2Face(x, y, z, rx, ry, rz float32, col color.NRGBA) Poly {
	return Poly{[4][3]float32{
		{z + rz, y - ry, x - rx},
		{z + rz, y + ry, x - rx},
		{z + rz, y + ry, x + rx},
		{z + rz, y - ry, x + rx}}, col}
}

func Load(fname string) {
	core.Log("loading", fname)
	f, err := os.Open(fname)
	core.Fatal(err)
	defer f.Close()
	r := dump.NewReader(f, dump.CRC_ENABLED)
	core.Fatal(r.Read())
	core.Log("loaded", fname)
	Frame = &(r.Frame)
	Crop2 = Frame.MeshSize
}

func PreRender() {
	core.Log("pre-render")
	polys = make([]Poly, 0, 10000)

	frame := Frame
	N = frame.MeshSize
	cell := frame.MeshStep
	maxworld := 0.
	for i := range N {
		world := float64(N[i]) * cell[i]
		if world > maxworld {
			maxworld = world
		}
	}
	scale := 1 / maxworld
	rx, ry, rz := float32(0.5*scale*cell[0]), float32(0.5*scale*cell[1]), float32(0.5*scale*cell[2])

	M := frame.Vectors()
	for i := Crop1[0]; i < Crop2[0]; i++ {
		x := float32(scale * cell[0] * (float64(i-N[0]/2) + 0.5))
		for j := Crop1[1]; j < Crop2[1]; j++ {
			y := float32(scale * cell[1] * (float64(j-N[1]/2) + 0.5))
			for k := Crop1[2]; k < Crop2[2]; k++ {
				z := float32(scale * cell[2] * (float64(k-N[2]/2) + 0.5))
				mx, my, mz := M[0][i][j][k], M[1][i][j][k], M[2][i][j][k]

				col := color.NRGBA{byte(0.5 * (mx + 1) * 255), byte(0.5 * (my + 1) * 255), byte(0.5 * (mz + 1) * 255), 255}
				// to be replaced, of course, by neighbor test
				if i == Crop1[0] {
					p := X1Face(x, y, z, rx, ry, rz, col)
					polys = append(polys, p)
				}
				if i == Crop2[0]-1 {
					p := X2Face(x, y, z, rx, ry, rz, col)
					polys = append(polys, p)
				}
				if j == Crop1[1] {
					p := Y1Face(x, y, z, rx, ry, rz, col)
					polys = append(polys, p)
				}
				if j == Crop2[1]-1 {
					p := Y2Face(x, y, z, rx, ry, rz, col)
					polys = append(polys, p)
				}
				if k == Crop1[2] {
					p := Z1Face(x, y, z, rx, ry, rz, col)
					polys = append(polys, p)
				}
				if k == Crop2[2]-1 {
					p := Z2Face(x, y, z, rx, ry, rz, col)
					polys = append(polys, p)
				}
			}
		}
	}
	core.Log("pre-rendered", len(polys), "polys")
}

func Render() {
	ClearScene()
	gl.Begin(gl.QUADS)
	for i := range polys {
		polys[i].Render()
	}
	gl.End()
	core.Log("rendered", len(polys), "polys")
}

func ClearScene() {
	ambient := []gl.Float{0.7, 0.7, 0.7, 1}
	diffuse := []gl.Float{1, 1, 1, 1}
	lightpos := []gl.Float{0.2, 0.5, 1, 1}
	gl.Lightfv(gl.LIGHT0, gl.AMBIENT, &ambient[0])
	gl.Lightfv(gl.LIGHT0, gl.DIFFUSE, &diffuse[0])
	gl.Lightfv(gl.LIGHT0, gl.POSITION, &lightpos[0])
	gl.Enable(gl.LIGHT0)
	gl.ClearColor(1, 1, 1, 1)
	gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
}
