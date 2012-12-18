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

func Load(fname string) *dump.Frame {
	core.Log("loading", fname)
	f, err := os.Open(fname)
	core.Fatal(err)
	defer f.Close()
	r := dump.NewReader(f, dump.CRC_ENABLED)
	core.Fatal(r.Read())
	core.Log("loaded", fname)
	return &(r.Frame)
}

func PreRender(frame *dump.Frame) []Poly {
	core.Log("pre-render")
	polys := make([]Poly, 0, 10000)

	size := frame.MeshSize
	N0, N1, N2 := size[0], size[1], size[2]
	cell := frame.MeshStep
	maxworld := 0.
	for i := range size {
		world := float64(size[i]) * cell[i]
		if world > maxworld {
			maxworld = world
		}
	}
	scale := 1 / maxworld
	rx, ry, rz := float32(0.5*scale*cell[0]), float32(0.5*scale*cell[1]), float32(0.5*scale*cell[2])

	M := frame.Vectors()
	for i := 0; i < N0; i++ {
		x := float32(scale * cell[0] * (float64(i-size[0]/2) + 0.5))
		for j := 0; j < N1; j++ {
			y := float32(scale * cell[1] * (float64(j-size[1]/2) + 0.5))
			for k := 0; k < N2; k++ {
				z := float32(scale * cell[2] * (float64(k-size[2]/2) + 0.5))
				mx, my, mz := M[0][i][j][k], M[1][i][j][k], M[2][i][j][k]

				col := color.NRGBA{byte(0.5 * (mx + 1) * 255), byte(0.5 * (my + 1) * 255), byte(0.5 * (mz + 1) * 255), 255}
				// to be replaced, of course, by neighbor test
				if i == 0 {
					p := X1Face(x, y, z, rx, ry, rz, col)
					polys = append(polys, p)
				}
				if i == N0-1 {
					p := X2Face(x, y, z, rx, ry, rz, col)
					polys = append(polys, p)
				}
				if j == 0 {
					p := Y1Face(x, y, z, rx, ry, rz, col)
					polys = append(polys, p)
				}
				if j == N1-1 {
					p := Y2Face(x, y, z, rx, ry, rz, col)
					polys = append(polys, p)
				}
				if k == 0 {
					p := Z1Face(x, y, z, rx, ry, rz, col)
					polys = append(polys, p)
				}
				if k == N2-1 {
					p := Z2Face(x, y, z, rx, ry, rz, col)
					polys = append(polys, p)
				}
			}
		}
	}
	core.Log("pre-rendered", len(polys), "polys")
	return polys
}

func Render(polys []Poly) {
	ClearScene()
	gl.Begin(gl.QUADS)
	for i := range polys {
		polys[i].Render()
	}
	gl.End()
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
