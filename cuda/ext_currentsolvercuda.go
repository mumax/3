package cuda

import (
	"github.com/mumax/3/data"
)

// Voltage solver

func CalculateVolt(m, voltage *data.Slice, ground ,terminal MSlice,mesh *data.Mesh) { 

	c := mesh.CellSize()
	N := mesh.Size()
	cfg := make3DConf(N)

	NN := mesh.Size()

	k_evaldvolt_async(
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		voltage.DevPtr(0),
		ground.DevPtr(0), ground.Mul(0),
		terminal.DevPtr(0), terminal.Mul(0),
		float32(c[X]), float32(c[Y]), float32(c[Z]), NN[X], NN[Y], NN[Z],
        	cfg)
}

func CalculateMaskJ(jmask,m, voltage *data.Slice,mesh *data.Mesh) { 

	c := mesh.CellSize()
	N := mesh.Size()
	cfg := make3DConf(N)

	NN := mesh.Size()

	k_calculatemaskJ2_async(
		jmask.DevPtr(X), jmask.DevPtr(Y), jmask.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		voltage.DevPtr(0),
		float32(c[X]), float32(c[Y]), float32(c[Z]), NN[X], NN[Y], NN[Z],
        	cfg)
	
	AcumJ:=0.0
	Cells:=0.0
	v:=0.0
	//print(NN[X],NN[Y],NN[Z],"\n")
	for i:=0;i<NN[X];i++{
		for j:=0;j<NN[Y];j++{
			for k:=0;k<NN[Z];k++{
				v=GetV(i,j,k,voltage)
				//print(v,"\n")
				if (v>=1.0) {
					AcumJ+=GetJ2(i,j,k,jmask)
					Cells++
				}
			}
		}
	}
	normJ:=AcumJ/Cells
	print (AcumJ,Cells,normJ,"\n")
    
    k_NormMaskJ_async(
		jmask.DevPtr(X), jmask.DevPtr(Y), jmask.DevPtr(Z),
		float32(normJ),
		NN[X], NN[Y], NN[Z],
        	cfg)
}


func GetJ2(ix, iy, iz int,Jmask *data.Slice) float64 {
	Jx:=GetCell(Jmask, 0, ix, iy, iz)
	Jy:=GetCell(Jmask, 1, ix, iy, iz)
	Jz:=GetCell(Jmask, 2, ix, iy, iz)
	return float64(Jx*Jx+Jy*Jy+Jz*Jz)
}

func GetV(ix, iy, iz int,volt *data.Slice) float64 {
	return float64(GetCell(volt, 0, ix, iy, iz))
}

