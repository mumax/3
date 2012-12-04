package render

type Light struct {
	Ambient [4]float32 // = []gl.Float{0.5, 0.5, 0.5, 1}
	Diffuse [4]float32 //  = []gl.Float{1, 1, 1, 1}
}
