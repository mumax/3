package engine

import ("math")

func init() {
	DeclFunc("ext_skxPos", skxPos, "Skyrmion core position (x,y) + radio (z)")
}

func skxPos(signo float64,skxlayer int) []float64 {

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
				if (m > 0)&&(signo>0)&&(z==skxlayer) {
					acumX+=float64(x)*float64(m)
					acumY+=float64(y)*float64(m)
                                        acumZ+=1
                                        acumM+=float64(m)
				}
				if (m < 0)&&(signo<0)&&(z==skxlayer) {
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






