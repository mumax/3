package engine

var CorePos = NewVectorValue("ext_corepos", "m", "Vortex core position (x,y) + polarization (z)", corePos)

func corePos() []float64 {
	m := M.Buffer()
	m_z := m.Comp(Z).HostCopy().Scalars()
	s := m.Size()
	Nx, Ny, Nz := s[X], s[Y], s[Z]

	max := float32(-1.0)
	var maxX, maxY, maxZ int

	for z := 0; z < Nz; z++ {
		// Avoid the boundaries so the neighbor interpolation can't go out of bounds.
		for y := 1; y < Ny-1; y++ {
			for x := 1; x < Nx-1; x++ {
				m := abs(m_z[z][y][x])
				if m > max {
					maxX, maxY, maxZ = x, y, z
					max = m
				}
			}
		}
	}

	pos := make([]float64, 3)
	mz := m_z[maxZ]

	// sub-cell interpolation in X and Y, but not Z
	pos[X] = float64(maxX) + interpolate_maxpos(
		max, -1, abs(mz[maxY][maxX-1]), 1, abs(mz[maxY][maxX+1])) -
		float64(Nx)/2 + 0.5
	pos[Y] = float64(maxY) + interpolate_maxpos(
		max, -1, abs(mz[maxY-1][maxX]), 1, abs(mz[maxY+1][maxX])) -
		float64(Ny)/2 + 0.5

	c := Mesh().CellSize()
	pos[X] *= c[X]
	pos[Y] *= c[Y]
	pos[Z] = float64(m_z[maxZ][maxY][maxX]) // 3rd coordinate is core polarization

	pos[X] += GetShiftPos() // add simulation window shift
	pos[Y] += GetShiftPosY()
	return pos
}

func interpolate_maxpos(f0, d1, f1, d2, f2 float32) float64 {
	b := (f2 - f1) / (d2 - d1)
	a := ((f2-f0)/d2 - (f0-f1)/(-d1)) / (d2 - d1)
	return float64(-b / (2 * a))
}

func abs(x float32) float32 {
	if x > 0 {
		return x
	} else {
		return -x
	}
}




//var SkxPos = NewVectorValue("ext_skxpos", "m", "Skyrmion core position (x,y) + radio (z)", skxPos)

func init() {
	DeclFunc("ext_skxPos", skxPos, "Skyrmion core position (x,y) + radio (z)")
}

func skxPos(signo float64) []float64 {

	m := M.Buffer()
	m_z := m.Comp(Z).HostCopy().Scalars()
	s := m.Size()
	Nx, Ny, Nz := s[X], s[Y], s[Z]

	acumX := float64(-0.0)
        acumY := float64(-0.0)
        acumZ := int(0)
        acumM := float64(0)

	for z := 0; z < Nz; z++ {
		// Avoid the boundaries so the neighbor interpolation can't go out of bounds.
		for y := 1; y < Ny-1; y++ {
			for x := 1; x < Nx-1; x++ {
				m := m_z[z][y][x]
				if (m > 0)&&(signo>0) {
					acumX+=float64(x)*float64(m)
					acumY+=float64(y)*float64(m)
                                        acumZ+=1
                                        acumM+=float64(m)
				}
				if (m < 0)&&(signo<0) {
					acumX+=float64(x)*float64(m)
					acumY+=float64(y)*float64(m)
                                        acumZ+=1
                                        acumM+=float64(m)
				}
			}
		}
	}

	pos := make([]float64, 3)

	c := Mesh().CellSize()
	pos[X] = c[X]*acumX/acumM
	pos[Y] = c[Y]*acumY/acumM
	pos[Z] = math.Pow(c[X]*c[Y]*float64(acumZ)/float64(3.1416),0.5)

	pos[X] += GetShiftPos() // add simulation window shift
        pos[Y] += GetShiftPosY() // add simulation window shift
	return pos
}

