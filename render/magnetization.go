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

func Load(fname string) *dump.Frame {
	f, err := os.Open(fname)
	core.Fatal(err)
	defer f.Close()
	r := dump.NewReader(f, dump.CRC_ENABLED)
	core.Fatal(r.Read())
	return &(r.Frame)
}

func PreRender(frame *dump.Frame) []Poly {
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
	for i := N0; i < N0; i++ {
		x := float32(scale * cell[0] * (float64(i-size[0]/2) + 0.5))
		for j := N1; j < N1; j++ {
			y := float32(scale * cell[1] * (float64(j-size[1]/2) + 0.5))
			for k := N2; k < N2; k++ {
				z := float32(scale * cell[2] * (float64(k-size[2]/2) + 0.5))
				mx, my, mz := M[0][i][j][k], M[1][i][j][k], M[2][i][j][k]

			}
		}
	}
	return polys
}

//func Render(frame *dump.Frame) {
//	ClearScene()
//
//	size := frame.MeshSize
//	cell := frame.MeshStep
//	maxworld := 0.
//	for i := range size {
//		world := float64(size[i]) * cell[i]
//		if world > maxworld {
//			maxworld = world
//		}
//	}
//	scale := 1 / maxworld
//	rx, ry, rz := float32(0.5*scale*cell[0]), float32(0.5*scale*cell[1]), float32(0.5*scale*cell[2])
//
//	gl.Color3f(0.5, 0.5, 0.5)
//	//Cube(0, 0, 0, 0.5, 0.5, 0.5)
//	M := frame.Vectors()
//	cubes := 0
//	for i := Min[0]; i < Max[0]; i++ {
//		x := float32(scale * cell[0] * (float64(i-size[0]/2) + 0.5))
//		for j := Min[1]; j < Max[1]; j++ {
//			y := float32(scale * cell[1] * (float64(j-size[1]/2) + 0.5))
//			for k := Min[2]; k < Max[2]; k++ {
//				z := float32(scale * cell[2] * (float64(k-size[2]/2) + 0.5))
//				mx, my, mz := M[0][i][j][k], M[1][i][j][k], M[2][i][j][k]
//				if i == Min[0] || i == Max[0]-1 ||
//					j == Min[1] || j == Max[1]-1 ||
//					k == Min[2] || k == Max[2]-1 {
//					gl.Color3f(gl.Float(0.5*(mx+0.5)), gl.Float(0.5*(my+0.5)), gl.Float(0.5*(mz+0.5)))
//					Cube(z, y, x, rz, ry, rx)
//					cubes++
//				}
//			}
//		}
//	}
//	core.Println(cubes, " cubes")
//}
//
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
