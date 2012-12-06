package render

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/dump"
	gl "github.com/chsc/gogl/gl21"
	"os"
)

func Load(fname string) *dump.Frame {
	f, err := os.Open(fname)
	core.Fatal(err)
	defer f.Close()

	r := dump.NewReader(f, dump.CRC_ENABLED)
	core.Fatal(r.Read())
	return &(r.Frame)
}

func Render(frame *dump.Frame) {
	ClearScene()

	ambdiff := []gl.Float{0.5, 0.5, 0.5, 1}
	gl.Materialfv(gl.FRONT_AND_BACK, gl.AMBIENT_AND_DIFFUSE, &ambdiff[0])

	N0, N1, N2 := frame.MeshSize[0], frame.MeshSize[1], frame.MeshSize[2]
	cell := frame.MeshStep
	scale := 1e299
	for i := 0; i < 3; i++ {
		s := 1 / (float64(frame.MeshSize[i]) * cell[i])
		if s < scale {
			scale = s
		}
	}
	rx := float32(scale * cell[0])
	ry := float32(scale * cell[1])
	rz := float32(scale * cell[2])
	m := frame.Vectors()
	for i := range m[0] {
		x := float32(scale * (float64(i-N0/2) + 0.5) * cell[0])
		for j := range m[0][i] {
			y := float32(scale * (float64(j-N1/2) + 0.5) * cell[1])
			for k := range m[0][i][j] {
				z := float32(scale * (float64(k-N2/2) + 0.5) * cell[2])
				(&Cube{Vertex{x, y, z}, Vertex{rx, ry, rz}}).Render()
				//log.Println(&Cube{Vertex{x, y, z}, Vertex{rx, ry, rz}})
			}
		}
	}

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
