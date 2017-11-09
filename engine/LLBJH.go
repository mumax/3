package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/util"
	"math"
)


// Heun solver for LLB equation + joule heating.
type HeunLLBJH struct{}

// Adaptive HeunLLBJH method, can be used as solver.Step
func (_ *HeunLLBJH) Step() {

	y := M.Buffer()
	dy0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy0)
	Hth1 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(Hth1)
	Hth2 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(Hth2)

	// Parameter for JH
	temp0 := TempJH.temp
	dtemp0 := cuda.Buffer(1, temp0.Size())
	defer cuda.Recycle(dtemp0)
	Kth := Kthermal.MSlice()
	defer Kth.Recycle()
	Cth := Cthermal.MSlice()
	defer Cth.Recycle()
	Dth := Density.MSlice()
	defer Dth.Recycle()
	Tsubsth := TSubs.MSlice()
	defer Tsubsth.Recycle()
	Tausubsth := TauSubs.MSlice()
	defer Tausubsth.Recycle()
	res := Resistivity.MSlice()
	defer res.Recycle()
	Qext := Qext.MSlice()
	defer Qext.Recycle()
	j := J.MSlice()
	defer j.Recycle()


	if FixDt != 0 {
		Dt_si = FixDt
	}

	dt := float32(Dt_si * GammaLL)
	util.Assert(dt > 0)

	// stage 1

        // Rewrite to calculate m step 1 
	torqueFnLLB(dy0,Hth1,Hth2)
	cuda.Madd2(y, y, dy0, 1, dt) // y = y + dt * dy
        cuda.Evaldt0(temp0,dtemp0,y,Kth,Cth,Dth,Tsubsth,Tausubsth,res,Qext,j,M.Mesh())
	cuda.Madd2(temp0, temp0, dtemp0, 1, dt/float32(GammaLL)) // temp = temp + dt * dtemp0
        

	// stage 2
	dy := cuda.Buffer(3, y.Size())
	defer cuda.Recycle(dy)

	dtemp := cuda.Buffer(1, dtemp0.Size())
	defer cuda.Recycle(dtemp)
	Time += Dt_si

        // Rewrite to calculate spep 2
	torqueFnLLB(dy,Hth1,Hth2)
        cuda.Evaldt0(temp0,dtemp,y,Kth,Cth,Dth,Tsubsth,Tausubsth,res,Qext,j,M.Mesh())

	err := cuda.MaxVecDiff(dy0, dy) * float64(dt)

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		cuda.Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt) //****
		cuda.Madd3(temp0, temp0, dtemp, dtemp0, 1, 0.5*dt/float32(GammaLL), -0.5*dt/float32(GammaLL)) //****
		//M.normalize()   // avoid it!!
		NSteps++
		adaptDt(math.Pow(MaxErr/err, 1./2.))
		setLastErr(err)
		setMaxTorque(dy)
	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time -= Dt_si
		cuda.Madd2(y, y, dy0, 1, -dt)  //****
		cuda.Madd2(temp0, temp0, dtemp0, 1, -dt/float32(GammaLL)) // temp = temp - dt * dtemp0
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./3.))
	}
}

func (_ *HeunLLBJH) Free() {}
