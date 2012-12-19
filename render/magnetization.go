package render

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/draw"
	gl "github.com/chsc/gogl/gl21"
	"github.com/jteeuwen/glfw"
)

var polys = make([]Poly, 10000)

func PreRender() {
	core.Log("pre-render")
	polys = polys[:0]

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

				col := draw.HSLMap(mz, my, mx)
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
}

func Render() {
	UpdateViewpos()
	ClearScene()
	gl.Begin(gl.QUADS)
	for i := range polys {
		polys[i].Render()
	}
	gl.End()
	glfw.SwapBuffers()
}

func ClearScene() {
	ambient := []gl.Float{0.7, 0.7, 0.7, 1}
	diffuse := []gl.Float{0.9, 0.9, 0.9, 1}
	lightpos := []gl.Float{0.2, 0.5, 1, 1}
	gl.Lightfv(gl.LIGHT0, gl.AMBIENT, &ambient[0])
	gl.Lightfv(gl.LIGHT0, gl.DIFFUSE, &diffuse[0])
	gl.Lightfv(gl.LIGHT0, gl.POSITION, &lightpos[0])
	gl.Enable(gl.LIGHT0)
	gl.ClearColor(1, 1, 1, 1)
	gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
}
