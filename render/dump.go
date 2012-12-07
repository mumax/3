package render

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/dump"
	gl "github.com/chsc/gogl/gl21"
	"os"
	"log"
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

	size := frame.MeshSize
	log.Println("size:", size)
	cell := frame.MeshStep
	log.Println("cell:", cell)
	maxworld := 0.
	var world [3]float64
	for i:=range world{
		world[i] = float64(size[i]) * cell[i]
		if world[i] > maxworld{maxworld = world[i]}
	}
	log.Println("world:", world)
	log.Println("maxworld:", maxworld)
	var scale [3]float64
	for i:=range scale{
		scale[i] = cell[i] / maxworld
	}
	log.Println("scale:", scale)

	N0, N1, N2 := float32(size[0]),  float32(size[1]),  float32(size[2]) 
	rx, ry, rz := 1./N0, 1./N1, 1./N2

	m := frame.Vectors()
	for i := range m[0] {
		x := float32(scale[0] * (float64(i-size[0]/2) + 0.5))
		for j := range m[0][i] {
			y := float32(scale[1] * (float64(j-size[1]/2) + 0.5))
			for k := range m[0][i][j] {
				z := float32(scale[2] * (float64(k-size[2]/2) + 0.5))
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
