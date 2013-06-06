package engine

func SetGeometry(s Shape) {
	geom = s
	regions.rasterGeom()
}
