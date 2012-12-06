package render

import(
 gl "github.com/chsc/gogl/gl21"
	"code.google.com/p/nimble-cube/dump"
	"code.google.com/p/nimble-cube/core"
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

func Render(data *dump.Frame) {
	ClearScene()

}

func ClearScene(){
	ambient := []gl.Float{0.7, 0.7, 0.7, 1}
	diffuse := []gl.Float{1, 1, 1, 1}
	lightpos := []gl.Float{0.2, 0.5, 1, 1}
	gl.Lightfv(gl.LIGHT0, gl.AMBIENT, &ambient[0])
	gl.Lightfv(gl.LIGHT0, gl.DIFFUSE, &diffuse[0])
	gl.Lightfv(gl.LIGHT0, gl.POSITION, &lightpos[0])
	gl.Enable(gl.LIGHT0)

	ambdiff := []gl.Float{0.5, 0.5, 0.0, 1}
	gl.Materialfv(gl.FRONT_AND_BACK, gl.AMBIENT_AND_DIFFUSE, &ambdiff[0])

	gl.ClearColor(1, 1, 1, 1)
	gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
}
