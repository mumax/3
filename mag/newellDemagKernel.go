package mag

import (
	"bufio"
	"fmt"
	"github.com/mumax/3/data"
	"github.com/mumax/3/oommf"
	"github.com/mumax/3/util"
	"math"
	"os"
)

/////////// Additional data structures ////////////////
const DEMAG_ASYMPTOTIC_DEFAULT_REFINE_RATIO float64 = 1.5

type DemagAsymptoticRefineData struct {
	rdx, rdy, rdz, result_scale float64
	xcount, ycount, zcount      int
}

type DemagNxxAsymptoticBase struct {
	cubic_cell                                                       bool
	self_demag, lead_weight                                          float64
	a1, a2, a3, a4, a5, a6                                           float64
	b1, b2, b3, b4, b5, b6, b7, b8, b9, b10                          float64
	c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15 float64
}

type DemagNxxAsymptotic struct {
	refine_data DemagAsymptoticRefineData
	Nxx         DemagNxxAsymptoticBase
}

type DemagNxyAsymptoticBase struct {
	cubic_cell                              bool
	lead_weight                             float64
	a1, a2, a3                              float64
	b1, b2, b3, b4, b5, b6                  float64
	c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 float64
}

type DemagNxyAsymptotic struct {
	refine_data DemagAsymptoticRefineData
	Nxy         DemagNxyAsymptoticBase
}

type DemagNxzAsymptotic struct {
	Nxy DemagNxyAsymptotic
}

type DemagNyzAsymptotic struct {
	Nxy DemagNxyAsymptotic
}

type DemagNyyAsymptotic struct {
	Nxx DemagNxxAsymptotic
}

type DemagNzzAsymptotic struct {
	Nxx DemagNxxAsymptotic
}

type DemagNabData struct {
	x, y, z       float64
	tx2, ty2, tz2 float64
	R, iR         float64
	R2, iR2       float64
}

////////////////// Subroutines ///////////////////////

///////// For working on new data structures /////////
func (s *DemagNabData) Set(import_x, import_y, import_z float64) { //OxsDemagNabData::Set
	s.x, s.y, s.z = import_x, import_y, import_z
	x2 := s.x * s.x
	y2 := s.y * s.y
	s.R2 = x2 + y2
	z2 := s.z * s.z
	s.R2 += z2
	R4 := s.R2 * s.R2
	s.R = math.Sqrt(s.R2)
	if s.R2 != 0.0 {
		s.tx2 = x2 / R4
		s.ty2 = y2 / R4
		s.tz2 = z2 / R4
		s.iR2 = float64(1.0 / s.R2)
		s.iR = float64(1.0 / s.R)
	} else { // Should never enter this if used correctly
		s.tx2, s.ty2, s.tz2 = 0.0, 0.0, 0.0
		s.iR2, s.R, s.iR = 0.0, 0.0, 0.0
	}
}

func DemagNabDataSetPair(ixa, iya, iza, ixb, iyb, izb float64, pta, ptb *DemagNabData) { //OxsDemagNabData::SetPair
	pta.Set(ixa, iya, iza)
	ptb.Set(ixb, iyb, izb)
}

// Need to call this to initialize the refine_data variables
func (s *DemagAsymptoticRefineData) Init(dx, dy, dz, maxratio float64) { // OxsDemagAsymptoticRefineData::OxsDemagAsymptoticRefineData
	s.rdx, s.rdy, s.rdz = 0.0, 0.0, 0.0
	s.result_scale = 0.0
	s.xcount, s.ycount, s.zcount = 0, 0, 0
	if dz <= dx && dz <= dy {
		xratio := math.Ceil(dx / (maxratio * dz))
		s.xcount = int(xratio)
		s.rdx = float64(dx / xratio)
		yratio := math.Ceil(dy / (maxratio * dz))
		s.ycount = int(yratio)
		s.rdy = float64(dy / yratio)
		s.zcount = 1
		s.rdz = dz
	} else if dy <= dx && dy <= dz {
		xratio := math.Ceil(dx / (maxratio * dy))
		s.xcount = int(xratio)
		s.rdx = float64(dx / xratio)
		zratio := math.Ceil(dz / (maxratio * dy))
		s.zcount = int(zratio)
		s.rdz = float64(dz / zratio)
		s.ycount = 1
		s.rdy = dy
	} else {
		yratio := math.Ceil(dy / (maxratio * dx))
		s.ycount = int(yratio)
		s.rdy = float64(dy / yratio)
		zratio := math.Ceil(dz / (maxratio * dx))
		s.zcount = int(zratio)
		s.rdz = float64(dz / zratio)
		s.xcount = 1
		s.rdx = dx
	}
	s.result_scale = float64(1.0) / (float64(s.xcount) * float64(s.ycount) * float64(s.zcount))
}

// Need to call this to initialize the Nxx, Nyy and Nzz variables
func (s *DemagNxxAsymptoticBase) Init(refine_data *DemagAsymptoticRefineData) { // Oxs_DemagNxxAsymptoticBase::Oxs_DemagNxxAsymptoticBase
	dx, dy, dz := refine_data.rdx, refine_data.rdy, refine_data.rdz
	s.self_demag = SelfDemagNx(dx, dy, dz)
	dx2, dy2, dz2 := dx*dx, dy*dy, dz*dz
	dx4, dy4, dz4 := dx2*dx2, dy2*dy2, dz2*dz2
	dx6, dy6, dz6 := dx4*dx2, dy4*dy2, dz4*dz2
	s.lead_weight = (float64(-1.0) * dx * dy * dz) / (float64(4.0) * math.Pi)
	// Initialize coefficients for 1/R^5 term
	if (dx2 != dy2) || (dx2 != dz2) || (dy2 != dz2) { // Non-cube case
		s.cubic_cell = false
		s.a1 = s.lead_weight / float64(4.0)
		s.a2, s.a3, s.a4, s.a5, s.a6 = s.a1, s.a1, s.a1, s.a1, s.a1
		s.a1 *= 8.0*dx2 - 4.0*dy2 - 4.0*dz2
		s.a2 *= -24.0*dx2 + 27.0*dy2 - 3.0*dz2
		s.a3 *= -24.0*dx2 - 3.0*dy2 + 27.0*dz2
		s.a4 *= 3.0*dx2 - 4.0*dy2 + dz2
		s.a5 *= 6.0*dx2 - 3.0*dy2 - 3.0*dz2
		s.a6 *= 3.0*dx2 + dy2 - 4.0*dz2
	} else { // Cube
		s.cubic_cell = true
		s.a1, s.a2, s.a3, s.a4, s.a5, s.a6 = float64(0.0), float64(0.0), float64(0.0), float64(0.0), float64(0.0), float64(0.0)
	}

	// Initialize coefficients for 1/R^7 term
	bInit := s.lead_weight / float64(16.0)
	s.b1, s.b2, s.b3, s.b4, s.b5, s.b6, s.b7, s.b8, s.b9, s.b10 = bInit, bInit, bInit, bInit, bInit, bInit, bInit, bInit, bInit, bInit
	if s.cubic_cell {
		s.b1 *= -14.0 * dx4
		s.b2 *= 105.0 * dx4
		s.b3 *= 105.0 * dx4
		s.b4 *= -105.0 * dx4
		s.b6 *= -105.0 * dx4
		s.b7 *= 7.0 * dx4
		s.b10 *= 7.0 * dx4
		s.b5, s.b8, s.b9 = float64(0.0), float64(0.0), float64(0.0)
	} else {
		s.b1 *= 32.0*dx4 - 40.0*dx2*dy2 - 40.0*dx2*dz2 + 12.0*dy4 + 10.0*dy2*dz2 + 12.0*dz4
		s.b2 *= -240.0*dx4 + 580.0*dx2*dy2 + 20.0*dx2*dz2 - 202.0*dy4 - 75.0*dy2*dz2 + 22.0*dz4
		s.b3 *= -240.0*dx4 + 20.0*dx2*dy2 + 580.0*dx2*dz2 + 22.0*dy4 - 75.0*dy2*dz2 - 202.0*dz4
		s.b4 *= 180.0*dx4 - 505.0*dx2*dy2 + 55.0*dx2*dz2 + 232.0*dy4 - 75.0*dy2*dz2 + 8.0*dz4
		s.b5 *= 360.0*dx4 - 450.0*dx2*dy2 - 450.0*dx2*dz2 - 180.0*dy4 + 900.0*dy2*dz2 - 180.0*dz4
		s.b6 *= 180.0*dx4 + 55.0*dx2*dy2 - 505.0*dx2*dz2 + 8.0*dy4 - 75.0*dy2*dz2 + 232.0*dz4
		s.b7 *= -10.0*dx4 + 30.0*dx2*dy2 - 5.0*dx2*dz2 - 16.0*dy4 + 10.0*dy2*dz2 - 2.0*dz4
		s.b8 *= -30.0*dx4 + 55.0*dx2*dy2 + 20.0*dx2*dz2 + 8.0*dy4 - 75.0*dy2*dz2 + 22.0*dz4
		s.b9 *= -30.0*dx4 + 20.0*dx2*dy2 + 55.0*dx2*dz2 + 22.0*dy4 - 75.0*dy2*dz2 + 8.0*dz4
		s.b10 *= -10.0*dx4 - 5.0*dx2*dy2 + 30.0*dx2*dz2 - 2.0*dy4 + 10.0*dy2*dz2 - 16.0*dz4
	}

	// Initialize coefficients for 1/R^9 term
	cInit := s.lead_weight / float64(192.0)
	s.c1, s.c2, s.c3, s.c4, s.c5, s.c6, s.c7, s.c8 = cInit, cInit, cInit, cInit, cInit, cInit, cInit, cInit
	s.c9, s.c10, s.c11, s.c12, s.c13, s.c14, s.c15 = cInit, cInit, cInit, cInit, cInit, cInit, cInit
	if s.cubic_cell {
		s.c1 *= 32.0 * dx6
		s.c2 *= -448.0 * dx6
		s.c3 *= -448.0 * dx6
		s.c4 *= -150.0 * dx6
		s.c5 *= 7620.0 * dx6
		s.c6 *= -150.0 * dx6
		s.c7 *= 314.0 * dx6
		s.c8 *= -3810.0 * dx6
		s.c9 *= -3810.0 * dx6
		s.c10 *= 314.0 * dx6
		s.c11 *= -16.0 * dx6
		s.c12 *= 134.0 * dx6
		s.c13 *= 300.0 * dx6
		s.c14 *= 134.0 * dx6
		s.c15 *= -16.0 * dx6
	} else {
		s.c1 *= 384.0*dx6 + -896.0*dx4*dy2 + -896.0*dx4*dz2 + 672.0*dx2*dy4 + 560.0*dx2*dy2*dz2 + 672.0*dx2*dz4 + -120.0*dy6 + -112.0*dy4*dz2 + -112.0*dy2*dz4 + -120.0*dz6
		s.c2 *= -5376.0*dx6 + 22624.0*dx4*dy2 + 2464.0*dx4*dz2 + -19488.0*dx2*dy4 + -7840.0*dx2*dy2*dz2 + 672.0*dx2*dz4 + 3705.0*dy6 + 2198.0*dy4*dz2 + 938.0*dy2*dz4 + -345.0*dz6
		s.c3 *= -5376.0*dx6 + 2464.0*dx4*dy2 + 22624.0*dx4*dz2 + 672.0*dx2*dy4 + -7840.0*dx2*dy2*dz2 + -19488.0*dx2*dz4 + -345.0*dy6 + 938.0*dy4*dz2 + 2198.0*dy2*dz4 + 3705.0*dz6
		s.c4 *= 10080.0*dx6 + -48720.0*dx4*dy2 + 1680.0*dx4*dz2 + 49770.0*dx2*dy4 + -2625.0*dx2*dy2*dz2 + -630.0*dx2*dz4 + -10440.0*dy6 + -1050.0*dy4*dz2 + 2100.0*dy2*dz4 + -315.0*dz6
		s.c5 *= 20160.0*dx6 + -47040.0*dx4*dy2 + -47040.0*dx4*dz2 + -6300.0*dx2*dy4 + 133350.0*dx2*dy2*dz2 + -6300.0*dx2*dz4 + 7065.0*dy6 + -26670.0*dy4*dz2 + -26670.0*dy2*dz4 + 7065.0*dz6
		s.c6 *= 10080.0*dx6 + 1680.0*dx4*dy2 + -48720.0*dx4*dz2 + -630.0*dx2*dy4 + -2625.0*dx2*dy2*dz2 + 49770.0*dx2*dz4 + -315.0*dy6 + 2100.0*dy4*dz2 + -1050.0*dy2*dz4 + -10440.0*dz6
		s.c7 *= -3360.0*dx6 + 17290.0*dx4*dy2 + -1610.0*dx4*dz2 + -19488.0*dx2*dy4 + 5495.0*dx2*dy2*dz2 + -588.0*dx2*dz4 + 4848.0*dy6 + -3136.0*dy4*dz2 + 938.0*dy2*dz4 + -75.0*dz6
		s.c8 *= -10080.0*dx6 + 32970.0*dx4*dy2 + 14070.0*dx4*dz2 + -6300.0*dx2*dy4 + -66675.0*dx2*dy2*dz2 + 12600.0*dx2*dz4 + -10080.0*dy6 + 53340.0*dy4*dz2 + -26670.0*dy2*dz4 + 3015.0*dz6
		s.c9 *= -10080.0*dx6 + 14070.0*dx4*dy2 + 32970.0*dx4*dz2 + 12600.0*dx2*dy4 + -66675.0*dx2*dy2*dz2 + -6300.0*dx2*dz4 + 3015.0*dy6 + -26670.0*dy4*dz2 + 53340.0*dy2*dz4 + -10080.0*dz6
		s.c10 *= -3360.0*dx6 + -1610.0*dx4*dy2 + 17290.0*dx4*dz2 + -588.0*dx2*dy4 + 5495.0*dx2*dy2*dz2 + -19488.0*dx2*dz4 + -75.0*dy6 + 938.0*dy4*dz2 + -3136.0*dy2*dz4 + 4848.0*dz6
		s.c11 *= 105.0*dx6 + -560.0*dx4*dy2 + 70.0*dx4*dz2 + 672.0*dx2*dy4 + -280.0*dx2*dy2*dz2 + 42.0*dx2*dz4 + -192.0*dy6 + 224.0*dy4*dz2 + -112.0*dy2*dz4 + 15.0*dz6
		s.c12 *= 420.0*dx6 + -1610.0*dx4*dy2 + -350.0*dx4*dz2 + 672.0*dx2*dy4 + 2345.0*dx2*dy2*dz2 + -588.0*dx2*dz4 + 528.0*dy6 + -3136.0*dy4*dz2 + 2198.0*dy2*dz4 + -345.0*dz6
		s.c13 *= 630.0*dx6 + -1470.0*dx4*dy2 + -1470.0*dx4*dz2 + -630.0*dx2*dy4 + 5250.0*dx2*dy2*dz2 + -630.0*dx2*dz4 + 360.0*dy6 + -1050.0*dy4*dz2 + -1050.0*dy2*dz4 + 360.0*dz6
		s.c14 *= 420.0*dx6 + -350.0*dx4*dy2 + -1610.0*dx4*dz2 + -588.0*dx2*dy4 + 2345.0*dx2*dy2*dz2 + 672.0*dx2*dz4 + -345.0*dy6 + 2198.0*dy4*dz2 + -3136.0*dy2*dz4 + 528.0*dz6
		s.c15 *= 105.0*dx6 + 70.0*dx4*dy2 + -560.0*dx4*dz2 + 42.0*dx2*dy4 + -280.0*dx2*dy2*dz2 + 672.0*dx2*dz4 + 15.0*dy6 + -112.0*dy4*dz2 + 224.0*dy2*dz4 + -192.0*dz6
	}
}

func (s *DemagNxxAsymptoticBase) NxxAsymptotic(ptdata *DemagNabData) float64 { // First of two functions to compute far field approx. to Nxx
	// Asymptotic approximation to Nxx term.
	if ptdata.iR2 <= 0.0 {
		return s.self_demag
	} // Asymptotic expansion doesn't apply for R==0. Fall back to self-demag calculation.

	tx2 := ptdata.tx2
	ty2 := ptdata.ty2
	tz2 := ptdata.tz2

	tz4 := tz2 * tz2
	tz6 := tz4 * tz2
	term3 := (2*tx2 - ty2 - tz2) * s.lead_weight
	term5 := float64(0.0)
	term7 := float64(0.0)

	if s.cubic_cell {
		ty4 := ty2 * ty2
		term7 = ((s.b1*tx2+(s.b2*ty2+s.b3*tz2))*tx2+(s.b4*ty4+s.b6*tz4))*tx2 + s.b7*ty4*ty2 + s.b10*tz6
	} else {
		term5 = (s.a1*tx2+(s.a2*ty2+s.a3*tz2))*tx2 + (s.a4*ty2+s.a5*tz2)*ty2 + s.a6*tz4
		term7 = ((s.b1*tx2+(s.b2*ty2+s.b3*tz2))*tx2+((s.b4*ty2+s.b5*tz2)*ty2+s.b6*tz4))*tx2 + ((s.b7*ty2+s.b8*tz2)*ty2+s.b9*tz4)*ty2 + s.b10*tz6
	}

	term9 := (((s.c1*tx2+(s.c2*ty2+s.c3*tz2))*tx2+((s.c4*ty2+s.c5*tz2)*ty2+s.c6*tz4))*tx2+(((s.c7*ty2+s.c8*tz2)*ty2+s.c9*tz4)*ty2+s.c10*tz6))*tx2 + (((s.c11*ty2+s.c12*tz2)*ty2+s.c13*tz4)*ty2+s.c14*tz6)*ty2 + s.c15*tz4*tz4

	Nxx := (term9 + term7 + term5 + term3) * ptdata.iR

	return Nxx
}

func (s *DemagNxxAsymptoticBase) NxxAsymptoticPair(ptA, ptB *DemagNabData) float64 { // Second of two functions to compute far field approx. to Nxx
	valA := s.NxxAsymptotic(ptA)
	valB := s.NxxAsymptotic(ptB)
	return valA + valB
}

func (s *DemagNxxAsymptotic) NxxAsymptotic(ptdata *DemagNabData) float64 { // Wrapper function to compute far field approx. to Nxx
	xcount := s.refine_data.xcount
	ycount := s.refine_data.ycount
	zcount := s.refine_data.zcount

	rdx := s.refine_data.rdx
	rdy := s.refine_data.rdy
	rdz := s.refine_data.rdz

	result_scale := s.refine_data.result_scale

	var rptdata, mrptdata DemagNabData
	zsum := float64(0.0)
	for k := 1 - zcount; k < zcount; k++ {
		zoff := ptdata.z + float64(k)*rdz
		ysum := float64(0.0)
		for j := 1 - ycount; j < ycount; j++ {
			// Compute interactions for x-strip
			yoff := ptdata.y + float64(j)*rdy
			rptdata.Set(ptdata.x, yoff, zoff)
			xsum := float64(xcount) * s.Nxx.NxxAsymptotic(&rptdata)
			for i := 1; i < xcount; i++ {
				rptdata.Set(ptdata.x+float64(i)*rdx, yoff, zoff)
				mrptdata.Set(ptdata.x-float64(i)*rdx, yoff, zoff)
				xsum += float64(xcount-i) * s.Nxx.NxxAsymptoticPair(&rptdata, &mrptdata)
			}
			// Weight x-strip interactions into xy-plate
			ysum += float64(ycount-int(math.Abs(float64(j)))) * xsum
		}
		// Weight xy-plate interactions into total sum
		zsum += float64(zcount-int(math.Abs(float64(k)))) * ysum
	}
	return zsum * result_scale
}

// Need to call this to initialize the Nxy, Nxz, and Nyz variables
func (s *DemagNxyAsymptoticBase) Init(refine_data *DemagAsymptoticRefineData) { // Oxs_DemagNxyAsymptoticBase::Oxs_DemagNxyAsymptoticBase
	dx, dy, dz := refine_data.rdx, refine_data.rdy, refine_data.rdz

	dx2 := dx * dx
	dy2 := dy * dy
	dz2 := dz * dz

	dx4 := dx2 * dx2
	dy4 := dy2 * dy2
	dz4 := dz2 * dz2

	dx6 := dx4 * dx2
	dy6 := dy4 * dy2
	dz6 := dz4 * dz2

	s.lead_weight = (-dx * dy * dz / (4 * math.Pi))

	// Initialize coefficients for 1/R^5 term
	if dx2 != dy2 || dx2 != dz2 || dy2 != dz2 { // Non-cube case
		s.cubic_cell = false
		aInit := (s.lead_weight * float64(5.0)) / float64(4.0)
		s.a1, s.a2, s.a3 = aInit, aInit, aInit
		s.a1 *= 4.0*dx2 - 3.0*dy2 - dz2
		s.a2 *= -3.0*dx2 + 4.0*dy2 - dz2
		s.a3 *= -3.0*dx2 - 3.0*dy2 + 6.0*dz2
	} else { // Cube
		s.cubic_cell = true
		s.a1, s.a2, s.a3 = float64(0.0), float64(0.0), float64(0.0)
	}

	// Initialize coefficients for 1/R^7 term
	bInit := (s.lead_weight * float64(7.0)) / float64(16.0)
	s.b1, s.b2, s.b3, s.b4, s.b5, s.b6 = bInit, bInit, bInit, bInit, bInit, bInit
	if s.cubic_cell {
		s.b1 *= -7.0 * dx4
		s.b2 *= 19.0 * dx4
		s.b3 *= 13.0 * dx4
		s.b4 *= -7.0 * dx4
		s.b5 *= 13.0 * dx4
		s.b6 *= -13.0 * dx4
	} else {
		s.b1 *= 16.0*dx4 - 30.0*dx2*dy2 - 10.0*dx2*dz2 + 10.0*dy4 + 5.0*dy2*dz2 + 2.0*dz4
		s.b2 *= -40.0*dx4 + 105.0*dx2*dy2 - 5.0*dx2*dz2 - 40.0*dy4 - 5.0*dy2*dz2 + 4.0*dz4
		s.b3 *= -40.0*dx4 - 15.0*dx2*dy2 + 115.0*dx2*dz2 + 20.0*dy4 - 35.0*dy2*dz2 - 32.0*dz4
		s.b4 *= 10.0*dx4 - 30.0*dx2*dy2 + 5.0*dx2*dz2 + 16.0*dy4 - 10.0*dy2*dz2 + 2.0*dz4
		s.b5 *= 20.0*dx4 - 15.0*dx2*dy2 - 35.0*dx2*dz2 - 40.0*dy4 + 115.0*dy2*dz2 - 32.0*dz4
		s.b6 *= 10.0*dx4 + 15.0*dx2*dy2 - 40.0*dx2*dz2 + 10.0*dy4 - 40.0*dy2*dz2 + 32.0*dz4
	}

	// Initialize coefficients for 1/R^9 term
	cInit := s.lead_weight / float64(64.0)
	s.c1, s.c2, s.c3, s.c4, s.c5, s.c6, s.c7, s.c8, s.c9, s.c10 = cInit, cInit, cInit, cInit, cInit, cInit, cInit, cInit, cInit, cInit
	if s.cubic_cell {
		s.c1 *= 48.0 * dx6
		s.c2 *= -142.0 * dx6
		s.c3 *= -582.0 * dx6
		s.c4 *= -142.0 * dx6
		s.c5 *= 2840.0 * dx6
		s.c6 *= -450.0 * dx6
		s.c7 *= 48.0 * dx6
		s.c8 *= -582.0 * dx6
		s.c9 *= -450.0 * dx6
		s.c10 *= 180.0 * dx6
	} else {
		s.c1 *= 576.0*dx6 + -2016.0*dx4*dy2 + -672.0*dx4*dz2 + 1680.0*dx2*dy4 + 840.0*dx2*dy2*dz2 + 336.0*dx2*dz4 + -315.0*dy6 + -210.0*dy4*dz2 + -126.0*dy2*dz4 + -45.0*dz6
		s.c2 *= -3024.0*dx6 + 13664.0*dx4*dy2 + 448.0*dx4*dz2 + -12670.0*dx2*dy4 + -2485.0*dx2*dy2*dz2 + 546.0*dx2*dz4 + 2520.0*dy6 + 910.0*dy4*dz2 + 84.0*dy2*dz4 + -135.0*dz6
		s.c3 *= -3024.0*dx6 + 1344.0*dx4*dy2 + 12768.0*dx4*dz2 + 2730.0*dx2*dy4 + -10185.0*dx2*dy2*dz2 + -8694.0*dx2*dz4 + -945.0*dy6 + 1680.0*dy4*dz2 + 2394.0*dy2*dz4 + 1350.0*dz6
		s.c4 *= 2520.0*dx6 + -12670.0*dx4*dy2 + 910.0*dx4*dz2 + 13664.0*dx2*dy4 + -2485.0*dx2*dy2*dz2 + 84.0*dx2*dz4 + -3024.0*dy6 + 448.0*dy4*dz2 + 546.0*dy2*dz4 + -135.0*dz6
		s.c5 *= 5040.0*dx6 + -9940.0*dx4*dy2 + -13580.0*dx4*dz2 + -9940.0*dx2*dy4 + 49700.0*dx2*dy2*dz2 + -6300.0*dx2*dz4 + 5040.0*dy6 + -13580.0*dy4*dz2 + -6300.0*dy2*dz4 + 2700.0*dz6
		s.c6 *= 2520.0*dx6 + 2730.0*dx4*dy2 + -14490.0*dx4*dz2 + 420.0*dx2*dy4 + -7875.0*dx2*dy2*dz2 + 17640.0*dx2*dz4 + -945.0*dy6 + 3990.0*dy4*dz2 + -840.0*dy2*dz4 + -3600.0*dz6
		s.c7 *= -315.0*dx6 + 1680.0*dx4*dy2 + -210.0*dx4*dz2 + -2016.0*dx2*dy4 + 840.0*dx2*dy2*dz2 + -126.0*dx2*dz4 + 576.0*dy6 + -672.0*dy4*dz2 + 336.0*dy2*dz4 + -45.0*dz6
		s.c8 *= -945.0*dx6 + 2730.0*dx4*dy2 + 1680.0*dx4*dz2 + 1344.0*dx2*dy4 + -10185.0*dx2*dy2*dz2 + 2394.0*dx2*dz4 + -3024.0*dy6 + 12768.0*dy4*dz2 + -8694.0*dy2*dz4 + 1350.0*dz6
		s.c9 *= -945.0*dx6 + 420.0*dx4*dy2 + 3990.0*dx4*dz2 + 2730.0*dx2*dy4 + -7875.0*dx2*dy2*dz2 + -840.0*dx2*dz4 + 2520.0*dy6 + -14490.0*dy4*dz2 + 17640.0*dy2*dz4 + -3600.0*dz6
		s.c10 *= -315.0*dx6 + -630.0*dx4*dy2 + 2100.0*dx4*dz2 + -630.0*dx2*dy4 + 3150.0*dx2*dy2*dz2 + -3360.0*dx2*dz4 + -315.0*dy6 + 2100.0*dy4*dz2 + -3360.0*dy2*dz4 + 1440.0*dz6
	}
}

func (s *DemagNxyAsymptoticBase) NxyAsymptotic(ptdata *DemagNabData) float64 {
	// Asymptotic approximation to Nxy term.

	if ptdata.R2 <= 0.0 {
		// Asymptotic expansion doesn't apply for R == 0. Fall back
		// to self-demag calculation.
		return float64(0.0)
	}

	tx2 := ptdata.tx2
	ty2 := ptdata.ty2
	tz2 := ptdata.tz2

	term3 := float64(3.0) * s.lead_weight

	term5 := float64(0.0)

	if !s.cubic_cell {
		term5 = s.a1*tx2 + s.a2*ty2 + s.a3*tz2
	}

	tz4 := tz2 * tz2
	term7 := (s.b1*tx2+(s.b2*ty2+s.b3*tz2))*tx2 + (s.b4*ty2+s.b5*tz2)*ty2 + s.b6*tz4

	term9 := ((s.c1*tx2+(s.c2*ty2+s.c3*tz2))*tx2+((s.c4*ty2+s.c5*tz2)*ty2+s.c6*tz4))*tx2 + ((s.c7*ty2+s.c8*tz2)*ty2+s.c9*tz4)*ty2 + s.c10*tz4*tz2

	x := ptdata.x
	y := ptdata.y
	iR2 := ptdata.iR2
	iR := ptdata.iR
	iR5 := iR2 * iR2 * iR
	Nxy := (term9 + term7 + term5 + term3) * iR5 * x * y

	// Error should be of order 1/R^11

	return Nxy
}

func (s *DemagNxyAsymptoticBase) NxyAsymptoticPair(ptA, ptB *DemagNabData) float64 {
	valA := s.NxyAsymptotic(ptA)
	valB := s.NxyAsymptotic(ptB)
	return valA + valB
}

func (s *DemagNxyAsymptotic) NxyAsymptotic(ptdata *DemagNabData) float64 {
	// Presumably at least one of xcount, ycount, or zcount is 1, but this
	// fact is not used in the following code.

	// Alias data from refine_data structure
	xcount := s.refine_data.xcount
	ycount := s.refine_data.ycount
	zcount := s.refine_data.zcount

	rdx := s.refine_data.rdx
	rdy := s.refine_data.rdy
	rdz := s.refine_data.rdz

	result_scale := s.refine_data.result_scale

	var (
		rptdata, mrptdata DemagNabData
	)

	zsum := float64(0.0)
	for k := 1 - zcount; k < zcount; k++ {
		zoff := ptdata.z + float64(k)*rdz
		ysum := float64(0.0)
		for j := 1 - ycount; j < ycount; j++ {
			// Compute interactions for x-strip
			yoff := ptdata.y + float64(j)*rdy
			rptdata.Set(ptdata.x, yoff, zoff)
			xsum := float64(xcount) * s.Nxy.NxyAsymptotic(&rptdata)
			for i := 1; i < xcount; i++ {
				rptdata.Set(ptdata.x+float64(i)*rdx, yoff, zoff)
				mrptdata.Set(ptdata.x-float64(i)*rdx, yoff, zoff)
				xsum += float64(xcount-i) * s.Nxy.NxyAsymptoticPair(&rptdata, &mrptdata)
			}
			// Weight x-strip interactions into xy-plate
			ysum += float64(ycount-int(math.Abs(float64(j)))) * xsum
		}
		// Weight xy-plate interactions into total sum
		zsum += float64(zcount-int(math.Abs(float64(k)))) * ysum
	}

	return zsum * result_scale
}

func (s *DemagNxzAsymptotic) Init(dx, dy, dz, maxratio float64) {
	s.Nxy.refine_data.Init(dx, dz, dy, maxratio)
	s.Nxy.Nxy.Init(&s.Nxy.refine_data)
}

func (s *DemagNxzAsymptotic) NxzAsymptotic(x, y, z float64) float64 {
	var ptdata DemagNabData
	ptdata.Set(x, z, y)
	return s.Nxy.NxyAsymptotic(&ptdata)
}

func (s *DemagNyzAsymptotic) Init(dx, dy, dz, maxratio float64) {
	s.Nxy.refine_data.Init(dy, dz, dx, maxratio)
	s.Nxy.Nxy.Init(&s.Nxy.refine_data)
}

func (s *DemagNyzAsymptotic) NyzAsymptotic(x, y, z float64) float64 {
	var ptdata DemagNabData
	ptdata.Set(y, z, x)
	return s.Nxy.NxyAsymptotic(&ptdata)
}

func (s *DemagNyyAsymptotic) Init(dx, dy, dz, maxratio float64) {
	s.Nxx.refine_data.Init(dy, dx, dz, maxratio)
	s.Nxx.Nxx.Init(&s.Nxx.refine_data)
}

func (s *DemagNyyAsymptotic) NyyAsymptotic(x, y, z float64) float64 {
	var ptdata DemagNabData
	ptdata.Set(y, x, z)
	return s.Nxx.NxxAsymptotic(&ptdata)
}

func (s *DemagNzzAsymptotic) Init(dx, dy, dz, maxratio float64) {
	s.Nxx.refine_data.Init(dz, dy, dx, maxratio)
	s.Nxx.Nxx.Init(&s.Nxx.refine_data)
}

func (s *DemagNzzAsymptotic) NzzAsymptotic(x, y, z float64) float64 {
	var ptdata DemagNabData
	ptdata.Set(z, y, x)
	return s.Nxx.NxxAsymptotic(&ptdata)
}

// For accurate summing
func accSum(n int, arr *[]float64) float64 {
	// Quick sort the values
	tmp0 := *arr
	qsort(n, arr)
	sum, corr := tmp0[n-1], float64(0.0)
	for idx := 1; idx < n; idx++ {
		i := n - 1 - idx
		x := tmp0[i]
		y := corr + x
		tmp := y - corr
		u := x - tmp
		t := y + sum
		tmp = t - sum
		v := y - tmp
		z := u + v
		sum = t + z
		tmp = sum - t
		corr = z - tmp
	}
	return sum
}

// Special qsort function to sort array by magnitude
func qsort(n int, arr *[]float64) {
	// n = number of elements to sort in arr (beginning from the front)
	// arr = pointer to array to be sorted

	// Do quick check for sorting small arrays
	if n <= 1 {
		return
	} // Do nothing if n is <= 1

	tmp0 := *arr

	if n == 2 { // If there are only 2 items..
		if math.Abs(tmp0[1]) < math.Abs(tmp0[0]) { // Swap if magnitudes are not sorted
			tmp1 := tmp0[1]
			tmp0[1] = tmp0[0]
			tmp0[0] = tmp1
		} else {
			return
		}
	}

	// Set pivot
	pvt := tmp0[n-1]
	pvtAbs := math.Abs(pvt)

	// Set up two bins for sorting
	var (
		botArr, topArr []float64
		botCnt, topCnt int
	)

	botCnt = 0
	topCnt = 0

	// Go through the portion of array that needs to be sorted
	for idx := 0; idx < (n - 1); idx++ {
		if pvtAbs > math.Abs(tmp0[idx]) {
			botCnt++
			botArr = append(botArr, tmp0[idx])
		} else {
			topCnt++
			topArr = append(topArr, tmp0[idx])
		}
	}

	// By now, the bottom bin contains elements smaller than pivot
	// and the top bin contains elements larger than or equal to pivot.
	// Recurse through the bins if they are not empty

	if botCnt > 1 {
		qsort(botCnt, &botArr)
	}
	if topCnt > 1 {
		qsort(topCnt, &topArr)
	}

	// All bins are sorted by now. All we have to do is go through the array
	// and place elements in there accordingly
	idx := 0 // Setup index for filling array with sorted values (treat as stack)

	// Fill array with elements from bottom bin if bottom bin is not empty
	if botCnt != 0 {
		for idx0 := 0; idx0 < botCnt; idx0++ {
			tmp0[idx] = botArr[idx0]
			idx++
		}
	}

	// Place pivot in proper place in array and update pointer to top of stack
	tmp0[idx] = pvt
	idx++

	// Fill array with elements from top bin if top bin is not empty
	if topCnt != 0 {
		for idx0 := 0; idx0 < topCnt; idx0++ {
			tmp0[idx] = topArr[idx0]
			idx++
		}
	}
	return
}

// This is an upoptimized implementation where we will go through each and every cell to
// calculate the tensor. Hence, we need some additional functions
func CalculateSDA00(x, y, z, dx, dy, dz float64) float64 {
	result := float64(0.0)
	if (x == 0.0) && (y == 0.0) && (z == 0.0) {
		result = SelfDemagNx(dx, dy, dz) * (4.0 * math.Pi * dx * dy * dz)
	} else {
		var arr []float64

		arr = append(arr, -1.0*Newell_f(x+dx, y+dy, z+dz))
		arr = append(arr, -1.0*Newell_f(x+dx, y-dy, z+dz))
		arr = append(arr, -1.0*Newell_f(x+dx, y-dy, z-dz))
		arr = append(arr, -1.0*Newell_f(x+dx, y+dy, z-dz))
		arr = append(arr, -1.0*Newell_f(x-dx, y+dy, z-dz))
		arr = append(arr, -1.0*Newell_f(x-dx, y+dy, z+dz))
		arr = append(arr, -1.0*Newell_f(x-dx, y-dy, z+dz))
		arr = append(arr, -1.0*Newell_f(x-dx, y-dy, z-dz))

		arr = append(arr, 2.0*Newell_f(x, y-dy, z-dz))
		arr = append(arr, 2.0*Newell_f(x, y-dy, z+dz))
		arr = append(arr, 2.0*Newell_f(x, y+dy, z+dz))
		arr = append(arr, 2.0*Newell_f(x, y+dy, z-dz))
		arr = append(arr, 2.0*Newell_f(x+dx, y+dy, z))
		arr = append(arr, 2.0*Newell_f(x+dx, y, z+dz))
		arr = append(arr, 2.0*Newell_f(x+dx, y, z-dz))
		arr = append(arr, 2.0*Newell_f(x+dx, y-dy, z))
		arr = append(arr, 2.0*Newell_f(x-dx, y-dy, z))
		arr = append(arr, 2.0*Newell_f(x-dx, y, z+dz))
		arr = append(arr, 2.0*Newell_f(x-dx, y, z-dz))
		arr = append(arr, 2.0*Newell_f(x-dx, y+dy, z))

		arr = append(arr, -4.0*Newell_f(x, y-dy, z))
		arr = append(arr, -4.0*Newell_f(x, y+dy, z))
		arr = append(arr, -4.0*Newell_f(x, y, z-dz))
		arr = append(arr, -4.0*Newell_f(x, y, z+dz))
		arr = append(arr, -4.0*Newell_f(x+dx, y, z))
		arr = append(arr, -4.0*Newell_f(x-dx, y, z))

		arr = append(arr, 8.0*Newell_f(x, y, z))
		result = accSum(27, &arr)
	}
	return result
}

func CalculateSDA11(x, y, z, dx, dy, dz float64) float64 {
	return CalculateSDA00(y, x, z, dy, dx, dz)
}

func CalculateSDA22(x, y, z, dx, dy, dz float64) float64 {
	return CalculateSDA00(z, y, x, dz, dy, dx)
}

func CalculateSDA01(x, y, z, l, h, e float64) float64 {
	if (x == 0.0) || (y == 0.0) {
		return float64(0.0)
	}

	var arr []float64

	arr = append(arr, -1.0*Newell_g(x-l, y-h, z-e))
	arr = append(arr, -1.0*Newell_g(x-l, y-h, z+e))
	arr = append(arr, -1.0*Newell_g(x+l, y-h, z+e))
	arr = append(arr, -1.0*Newell_g(x+l, y-h, z-e))
	arr = append(arr, -1.0*Newell_g(x+l, y+h, z-e))
	arr = append(arr, -1.0*Newell_g(x+l, y+h, z+e))
	arr = append(arr, -1.0*Newell_g(x-l, y+h, z+e))
	arr = append(arr, -1.0*Newell_g(x-l, y+h, z-e))

	arr = append(arr, 2.0*Newell_g(x, y+h, z-e))
	arr = append(arr, 2.0*Newell_g(x, y+h, z+e))
	arr = append(arr, 2.0*Newell_g(x, y-h, z+e))
	arr = append(arr, 2.0*Newell_g(x, y-h, z-e))
	arr = append(arr, 2.0*Newell_g(x-l, y-h, z))
	arr = append(arr, 2.0*Newell_g(x-l, y+h, z))
	arr = append(arr, 2.0*Newell_g(x-l, y, z-e))
	arr = append(arr, 2.0*Newell_g(x-l, y, z+e))
	arr = append(arr, 2.0*Newell_g(x+l, y, z+e))
	arr = append(arr, 2.0*Newell_g(x+l, y, z-e))
	arr = append(arr, 2.0*Newell_g(x+l, y-h, z))
	arr = append(arr, 2.0*Newell_g(x+l, y+h, z))

	arr = append(arr, -4.0*Newell_g(x-l, y, z))
	arr = append(arr, -4.0*Newell_g(x+l, y, z))
	arr = append(arr, -4.0*Newell_g(x, y, z+e))
	arr = append(arr, -4.0*Newell_g(x, y, z-e))
	arr = append(arr, -4.0*Newell_g(x, y-h, z))
	arr = append(arr, -4.0*Newell_g(x, y+h, z))

	arr = append(arr, 8.0*Newell_g(x, y, z))

	return accSum(27, &arr)
}

func CalculateSDA02(x, y, z, dx, dy, dz float64) float64 {
	return CalculateSDA01(x, z, y, dx, dz, dy)
}

func CalculateSDA12(x, y, z, dx, dy, dz float64) float64 {
	return CalculateSDA01(y, z, x, dy, dz, dx)
}

// Newell's f(x, y, z) function
func Newell_f(x, y, z float64) float64 {
	x = math.Abs(x)
	xsq := x * x
	y = math.Abs(y)
	ysq := y * y
	z = math.Abs(z)
	zsq := z * z

	Rsq := xsq + ysq + zsq
	if Rsq <= 0.0 {
		return float64(0.0)
	}
	R := math.Sqrt(Rsq)

	var piece []float64
	piececount := int(0)
	if z > 0.0 {
		var temp1 float64
		var temp2 float64
		var temp3 float64
		piece = append(piece, float64(2.0*(2.0*xsq-ysq-zsq)*R))
		piececount++
		temp1 = x * y * z
		if temp1 > 0.0 {
			piece = append(piece, float64(-12.0*temp1*math.Atan2(y*z, x*R)))
			piececount++
		}
		temp2 = xsq + zsq
		if (y > 0.0) && (temp2 > 0.0) {
			dummy := math.Log(((y + R) * (y + R)) / temp2)
			piece = append(piece, float64(3.0*y*(zsq-xsq)*dummy))
			piececount++
		}
		temp3 = xsq + ysq
		if temp3 > 0.0 {
			dummy := math.Log(((z + R) * (z + R)) / temp3)
			piece = append(piece, float64(3.0*z*(ysq-xsq)*dummy))
			piececount++
		}
	} else {
		if x == y {
			K := 2.0*math.Sqrt(2.0) - 6.0*math.Log(1.0+math.Sqrt(2.0))
			piece = append(piece, float64(K*xsq*x))
			piececount++
		} else {
			piece = append(piece, float64(2.0*(2.0*xsq-ysq)*R))
			piececount++
			if (y > 0.0) && (x > 0.0) {
				piece = append(piece, float64(-6.0*y*xsq*math.Log((y+R)/x)))
				piececount++
			}
		}
	}

	return float64(accSum(piececount, &piece) / 12.0)
}

// Newell's g(x, y, z) function
func Newell_g(x, y, z float64) float64 {
	result_sign := float64(1.0)
	if x < 0.0 {
		result_sign *= -1.0
	}
	if y < 0.0 {
		result_sign *= -1.0
	}
	x = math.Abs(x)
	xsq := x * x
	y = math.Abs(y)
	ysq := y * y
	z = math.Abs(z)
	zsq := z * z

	Rsq := xsq + ysq + zsq
	if Rsq <= 0.0 {
		return float64(0.0)
	}
	R := math.Sqrt(Rsq)

	var piece []float64
	piececount := int(0)
	piece = append(piece, float64(-2.0*x*y*R))
	piececount++

	if z > 0.0 {
		piece = append(piece, float64(-z*zsq*math.Atan2(x*y, z*R)))
		piececount++
		piece = append(piece, float64(-3.0*z*ysq*math.Atan2(x*z, y*R)))
		piececount++
		piece = append(piece, float64(-3.0*z*xsq*math.Atan2(y*z, x*R)))
		piececount++

		var temp1 float64
		var temp2 float64
		var temp3 float64

		temp1 = xsq + ysq
		if temp1 > 0.0 {
			piece = append(piece, float64(3.0*x*y*z*math.Log((z+R)*(z+R)/temp1)))
			piececount++
		}
		temp2 = ysq + zsq
		if temp2 > 0.0 {
			piece = append(piece, float64(0.5*y*(3.0*zsq-ysq)*math.Log((x+R)*(x+R)/temp2)))
			piececount++
		}
		temp3 = xsq + zsq
		if temp3 > 0.0 {
			piece = append(piece, float64(0.5*x*(3.0*zsq-xsq)*math.Log((y+R)*(y+R)/temp3)))
			piececount++
		}
	} else {
		if y > 0.0 {
			piece = append(piece, float64(-y*ysq*math.Log((x+R)/y)))
			piececount++
		}
		if x > 0.0 {
			piece = append(piece, float64(-x*xsq*math.Log((y+R)/x)))
			piececount++
		}
	}
	return float64(result_sign * accSum(piececount, &piece) / 6.0)
}

// Some subroutines for accurate self-demag computations
func SelfDemagNx(x, y, z float64) float64 {
	if (x <= 0.0) || (y <= 0.0) || (z <= 0.0) {
		return float64(0.0)
	}
	if (x == y) && (y == z) {
		return (float64(1.0) / float64(3.0))
	}
	xsq, ysq, zsq := x*x, y*y, z*z
	R := math.Sqrt(xsq + ysq + zsq)
	Rxy := math.Sqrt(xsq + ysq)
	Rxz := math.Sqrt(xsq + zsq)
	Ryz := math.Sqrt(ysq + zsq)

	var arr []float64

	tmp := float64(2.0 * x * y * z * ((x/(x+Rxy)+(2*xsq+ysq+zsq)/(R*Rxy+x*Rxz))/(x+Rxz) + (x/(x+Rxz)+(2*xsq+ysq+zsq)/(R*Rxz+x*Rxy))/(x+Rxy)) / ((x + R) * (Rxy + Rxz + R)))
	arr = append(arr, tmp)

	tmp = float64(-1.0 * x * y * z * ((y/(y+Rxy)+(2*ysq+xsq+zsq)/(R*Rxy+y*Ryz))/(y+Ryz) + (y/(y+Ryz)+(2*ysq+xsq+zsq)/(R*Ryz+y*Rxy))/(y+Rxy)) / ((y + R) * (Rxy + Ryz + R)))
	arr = append(arr, tmp)

	tmp = float64(-1.0 * x * y * z * ((z/(z+Rxz)+(2*zsq+xsq+ysq)/(R*Rxz+z*Ryz))/(z+Ryz) + (z/(z+Ryz)+(2*zsq+xsq+ysq)/(R*Ryz+z*Rxz))/(z+Rxz)) / ((z + R) * (Rxz + Ryz + R)))
	arr = append(arr, tmp)

	tmp = float64(6.0 * math.Atan(y*z/(x*R)))
	arr = append(arr, tmp)

	piece4 := float64(-1.0 * y * z * z * (1.0/(x+Rxz) + y/(Rxy*Rxz+x*R)) / (Rxz * (y + Rxy)))
	if piece4 > -0.5 {
		arr = append(arr, float64(3.0*x*math.Log1p(piece4)/z))
	} else {
		arr = append(arr, float64(3.0*x*math.Log(x*(y+R)/(Rxz*(y+Rxy)))/z))
	}

	piece5 := float64(-1.0 * y * y * z * (1.0/(x+Rxy) + z/(Rxy*Rxz+x*R)) / (Rxy * (z + Rxz)))
	if piece5 > -0.5 {
		arr = append(arr, float64(3.0*x*math.Log1p(piece5)/y))
	} else {
		arr = append(arr, float64(3.0*x*math.Log(x*(z+R)/(Rxy*(z+Rxz)))/y))
	}

	piece6 := float64(-1.0 * xsq * z * (1/(y+Rxy) + z/(Rxy*Ryz+y*R)) / (Rxy * (z + Ryz)))
	if piece6 > -0.5 {
		arr = append(arr, float64(-3.0*y*math.Log1p(piece6)/x))
	} else {
		arr = append(arr, float64(-3.0*y*math.Log(y*(z+R)/(Rxy*(z+Ryz)))/x))
	}

	piece7 := float64(-1.0 * xsq * y * (1/(z+Rxz) + y/(Rxz*Ryz+z*R)) / (Rxz * (y + Ryz)))
	if piece7 > -0.5 {
		arr = append(arr, float64(-3.0*z*math.Log1p(piece7)/x))
	} else {
		arr = append(arr, float64(-3.0*z*math.Log(z*(y+R)/(Rxz*(y+Ryz)))/x))
	}

	return accSum(8, &arr) / (3.0 * math.Pi)
}

func SelfDemagNy(xsize, ysize, zsize float64) float64 {
	return SelfDemagNx(ysize, zsize, xsize)
}

func SelfDemagNz(xsize, ysize, zsize float64) float64 {
	return SelfDemagNx(zsize, xsize, ysize)
}

//////////////////////////////////////////////////////////////////////////
// Greatest common divisor via Euclid's algorithm
func Gcd(m, n float64) float64 {
	util.Assert((m == math.Floor(m)) && (n == math.Floor(n)))
	m = math.Floor(math.Abs(m))
	n = math.Floor(math.Abs(n))
	if (m == 0.0) || (n == 0.0) {
		return 0.0
	}
	temp := m - math.Floor(m/n)*n
	for temp > 0.0 {
		m = n
		n = temp
		temp = m - math.Floor(m/n)*n
	}
	return n
}

//////////////////////////////////////////////////////////////////////////
// Simple continued fraction like rational approximator.  Unlike the usual
// continued fraction expansion, here we allow +/- on each term (rounds to
// nearest integer rather than truncating).  This converges faster (by
// producing fewer terms).
//   A more useful rat approx routine would probably take an error
// tolerance rather than a step count, but we'll leave that until the time
// when we have an actual use for this routine.  (There is a 4 term
// recursion relation for continued fractions in "Numerical Recipes in C"
// that can probably be used, and also remove the need for the coef[]
// array.)  -mjd, 23-Jan-2000. Takes x and y as separate imports, and also
// takes error tolerance imports. Fills exports p and q with best result,
// and returns 1 or 0 according to whether or not p/q meets the specified
// relative error.
func FindRatApprox(x float64, y float64, relerr float64, maxq float64, p *float64, q *float64) bool {
	sign := 1
	if x < 0.0 {
		x *= -1.0
		sign *= -1
	}
	if y < 0.0 {
		y *= -1.0
		sign *= -1
	}
	swap := false
	if x < y {
		t := x
		x = y
		y = t
		swap = true
	}

	x0 := x
	y0 := y

	p0, p1 := float64(0.0), float64(1.0)
	q0, q1 := float64(1.0), float64(0.0)

	m := math.Floor(x / y)
	r := x - m*y
	p2 := m*p1 + p0
	q2 := m*q1 + q0
	flag := (q2 < maxq) && (math.Abs(x0*q2-p2*y0) > relerr*x0*q2)
	for flag {
		x = y
		y = r
		m = math.Floor(x / y)
		r = x - m*y
		p0 = p1
		p1 = p2
		p2 = m*p1 + p0
		q0 = q1
		q1 = q2
		q2 = m*q1 + q0
		flag = (q2 < maxq) && (math.Abs(x0*q2-p2*y0) > relerr*x0*q2)
	}

	if !swap {
		*p = p2
		*q = q2
	} else {
		*p = q2
		*q = p2
	}

	flag = (math.Abs(x0*q2-p2*y0) <= relerr*x0*q2)
	return flag
}

func GetNextPowerOfTwo(n int) int {
	m := 1
	logsize := 0
	for m < n {
		m *= 2
		logsize += 1
		util.Assert(m > 0)
	}
	return m
}

/////////////////// Main Codes ////////////////////////

// Obtains the demag kernel either from cacheDir/ or by calculating (and then storing in cacheDir for next time).
// Empty cacheDir disables caching.
func NewellDemagKernel(inputSize, pbc [3]int, cellsize [3]float64, asymptotic_radius, zero_self_demag int, cacheDir string) (kernel [3][3]*data.Slice) {
	sanityCheck(cellsize, pbc)
	// Cache disabled
	if cacheDir == "" {
		util.Log(`Not using kernel cache (-cache="")`)
		return CalcNewellDemagKernel(inputSize, pbc, cellsize, asymptotic_radius, zero_self_demag)
	}

	// Error-resilient kernel cache: if anything goes wrong, return calculated kernel.
	defer func() {
		if err := recover(); err != nil {
			util.Log("Unable to use kernel cache:", err)
			kernel = CalcNewellDemagKernel(inputSize, pbc, cellsize, asymptotic_radius, zero_self_demag)
		}
	}()

	// Try to load kernel
	basename := fmt.Sprint(cacheDir, "/", "mumax3kernel_", inputSize, "_", pbc, "_", cellsize, "_", zero_self_demag, "_")
	var errLoad error
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			if inputSize[Z] == 1 && ((i == X && j == Z) || (i == Y && j == Z)) {
				continue // element not needed in 2D
			}
			kernel[i][j], errLoad = LoadKernel(fmt.Sprint(basename, i, j, ".ovf"))
			if errLoad != nil {
				break
			}
		}
		if errLoad != nil {
			break
		}
	}
	// make result symmetric for tools that expect it so.
	kernel[Y][X] = kernel[X][Y]
	kernel[Z][X] = kernel[X][Z]
	kernel[Z][Y] = kernel[Y][Z]

	if errLoad != nil {
		util.Log("Did not use cached kernel:", errLoad)
	} else {
		util.Log("Using cached kernel:", basename)
		return kernel
	}

	// Could not load kernel: calculate it and save
	var errSave error
	kernel = CalcNewellDemagKernel(inputSize, pbc, cellsize, asymptotic_radius, zero_self_demag)
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			if inputSize[Z] == 1 && ((i == X && j == Z) || (i == Y && j == Z)) {
				continue // element not needed in 2D
			}
			errSave = SaveKernel(fmt.Sprint(basename, i, j, ".ovf"), kernel[i][j])
			if errSave != nil {
				break
			}
		}
		if errSave != nil {
			break
		}
	}
	if errSave != nil {
		util.Log("Failed to cache kernel:", errSave)
	} else {
		util.Log("Cached kernel:", basename)
	}

	return kernel
}

func LoadNewellKernel(fname string) (kernel *data.Slice, err error) {
	kernel, _, err = oommf.ReadFile(fname)
	return
}

func SaveNewellKernel(fname string, kernel *data.Slice) error {
	f, err := os.OpenFile(fname, os.O_WRONLY|os.O_TRUNC|os.O_CREATE, 0666)
	if err != nil {
		return err
	}
	out := bufio.NewWriter(f)
	defer out.Flush()
	oommf.WriteOVF2(out, kernel, data.Meta{}, "binary 4")
	return nil
}

// Calculates the magnetostatic kernel by Newell's formulation. This is the same as the brtue force integration
// of magnetic charges over the faces and averages over cell volumes, except that the integration between discrete
// cells is given analytically by Newell's paper (just as used in OOMMF). This implementation is a port from OOMMF.
//
// size: size of the kernel, usually 2 x larger than the size of the magnetization due to zero padding
// zero_self_demag: To determine whether to set the self-demagnetization to zero
//
// return value: A symmetric rank 5 tensor K[sourcedir][destdir][x][y][z]
// (e.g. K[X][Y][1][2][3] gives H_y at position (1, 2, 3) due to a unit dipole m_x at the origin.
// You can use the function KernIdx to convert from source-dest pairs like XX to 1D indices:
// K[KernIdx[X][X]] returns K[XX]
//
func CalcNewellDemagKernel(inputSize, pbc [3]int, cellsize [3]float64, asymptotic_radius, zero_self_demag int) (kernel [3][3]*data.Slice) {

	// Add zero-padding in non-PBC directions
	size := padSize(inputSize, pbc)

	// Sanity check
	{
		util.Assert(size[Z] > 0 && size[Y] > 0 && size[X] > 0)
		util.Assert(cellsize[X] > 0 && cellsize[Y] > 0 && cellsize[Z] > 0)
		util.Assert(pbc[X] >= 0 && pbc[Y] >= 0 && pbc[Z] >= 0)
		util.Assert(zero_self_demag >= 0)
	}

	// Field (destination) loop ranges
	r1, r2 := kernelRanges(size, pbc)

	// Allocate only upper diagonal part. The rest is symmetric due to reciprocity.
	var array [3][3][][][]float32
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			kernel[i][j] = data.NewSlice(1, size)
			array[i][j] = kernel[i][j].Scalars()
		}
	}

	dx, dy, dz := cellsize[X], cellsize[Y], cellsize[Z]
	// Determine relative sizes of dx, dy and dz, since that is all that demag
	// calculation cares about

	var (
		p1, p2, q1, q2 float64
	)

	if FindRatApprox(dx, dy, 1e-12, 1000, &p1, &q1) && FindRatApprox(dz, dy, 1e-12, 1000, &p2, &q2) {
		gcd := Gcd(q1, q2)
		dx = p1 * q2 / gcd
		dy = q1 * q2 / gcd
		dz = p2 * q1 / gcd
	} else {
		maxedge := dx
		if dy > maxedge {
			maxedge = dy
		}
		if dz > maxedge {
			maxedge = dz
		}
		dx /= maxedge
		dy /= maxedge
		dz /= maxedge
	}

	numChans := int(r2[Z] - r1[Z] + 1)
	done := make(chan struct{}, numChans) // parallel calculation of one component done?

	scale := float64(-1.0 / (4.0 * math.Pi * dx * dy * dz)) // The subroutines return -N instead but N is expected in mumax
	scaled_arad := float64(asymptotic_radius) * math.Pow(float64(dx*dy*dz), float64(1.0/3.0))

	scaled_arad_sq := scaled_arad * scaled_arad

	for z := r1[Z]; z <= r2[Z]; z++ {
		go func(z int) {
			zw := wrap(z, size[Z])
			// skip one half, reconstruct from symmetry later
			// check on wrapped index instead of loop range so it also works for PBC
			if zw > size[Z]/2 {
			} else {
				var (
					R, R2 [3]float64 // field and source cell center positions
				)

				R[Z] = float64(z) * dz
				R2[Z] = R[Z] * R[Z]

				for y := r1[Y]; y <= r2[Y]; y++ {
					yw := wrap(y, size[Y])
					if yw > size[Y]/2 {
						continue
					}
					R[Y] = float64(y) * dy
					R2[Y] = R[Y] * R[Y]

					for x := r1[X]; x <= r2[X]; x++ {
						xw := wrap(x, size[X])
						if xw > size[X]/2 {
							continue
						}
						R[X] = float64(x) * dx
						R2[X] = R[X] * R[X]
						if scaled_arad >= 0.0 {
							testVal := scaled_arad_sq - R2[Z] - R2[Y]
							if R2[X] > testVal {
								ANxx := new(DemagNxxAsymptotic)
								ANxx.refine_data.Init(dx, dy, dz, DEMAG_ASYMPTOTIC_DEFAULT_REFINE_RATIO)
								ANxx.Nxx.Init(&ANxx.refine_data)

								ANxy := new(DemagNxyAsymptotic)
								ANxy.refine_data.Init(dx, dy, dz, DEMAG_ASYMPTOTIC_DEFAULT_REFINE_RATIO)
								ANxy.Nxy.Init(&ANxy.refine_data)

								ANxz := new(DemagNxzAsymptotic)
								ANxz.Init(dx, dy, dz, DEMAG_ASYMPTOTIC_DEFAULT_REFINE_RATIO)

								ANyy := new(DemagNyyAsymptotic)
								ANyy.Init(dx, dy, dz, DEMAG_ASYMPTOTIC_DEFAULT_REFINE_RATIO)

								ANyz := new(DemagNyzAsymptotic)
								ANyz.Init(dx, dy, dz, DEMAG_ASYMPTOTIC_DEFAULT_REFINE_RATIO)

								ANzz := new(DemagNzzAsymptotic)
								ANzz.Init(dx, dy, dz, DEMAG_ASYMPTOTIC_DEFAULT_REFINE_RATIO)

								var ptdata DemagNabData
								ptdata.Set(R[X], R[Y], R[Z])
								array[X][X][zw][yw][xw] += float32(-1.0 * ANxx.NxxAsymptotic(&ptdata))
								array[X][Y][zw][yw][xw] += float32(-1.0 * ANxy.NxyAsymptotic(&ptdata))
								array[X][Z][zw][yw][xw] += float32(-1.0 * ANxz.NxzAsymptotic(R[X], R[Y], R[Z]))
								array[Y][Y][zw][yw][xw] += float32(-1.0 * ANyy.NyyAsymptotic(R[X], R[Y], R[Z]))
								array[Y][Z][zw][yw][xw] += float32(-1.0 * ANyz.NyzAsymptotic(R[X], R[Y], R[Z]))
								array[Z][Z][zw][yw][xw] += float32(-1.0 * ANzz.NzzAsymptotic(R[X], R[Y], R[Z]))
							} else {
								array[X][X][zw][yw][xw] += float32(CalculateSDA00(R[X], R[Y], R[Z], dx, dy, dz) * scale) // Calculate Nxx
								array[X][Y][zw][yw][xw] += float32(CalculateSDA01(R[X], R[Y], R[Z], dx, dy, dz) * scale) // Calculate Nxy
								array[X][Z][zw][yw][xw] += float32(CalculateSDA02(R[X], R[Y], R[Z], dx, dy, dz) * scale) // Calculate Nxz
								array[Y][Y][zw][yw][xw] += float32(CalculateSDA11(R[X], R[Y], R[Z], dx, dy, dz) * scale) // Calculate Nyy
								array[Y][Z][zw][yw][xw] += float32(CalculateSDA12(R[X], R[Y], R[Z], dx, dy, dz) * scale) // Calculate Nyz
								array[Z][Z][zw][yw][xw] += float32(CalculateSDA22(R[X], R[Y], R[Z], dx, dy, dz) * scale) // Calculate Nzz
							}
						} else {
							array[X][X][zw][yw][xw] += float32(CalculateSDA00(R[X], R[Y], R[Z], dx, dy, dz) * scale) // Calculate Nxx
							array[X][Y][zw][yw][xw] += float32(CalculateSDA01(R[X], R[Y], R[Z], dx, dy, dz) * scale) // Calculate Nxy
							array[X][Z][zw][yw][xw] += float32(CalculateSDA02(R[X], R[Y], R[Z], dx, dy, dz) * scale) // Calculate Nxz
							array[Y][Y][zw][yw][xw] += float32(CalculateSDA11(R[X], R[Y], R[Z], dx, dy, dz) * scale) // Calculate Nyy
							array[Y][Z][zw][yw][xw] += float32(CalculateSDA12(R[X], R[Y], R[Z], dx, dy, dz) * scale) // Calculate Nyz
							array[Z][Z][zw][yw][xw] += float32(CalculateSDA22(R[X], R[Y], R[Z], dx, dy, dz) * scale) // Calculate Nzz
						}
					}
				}
			}
			done <- struct{}{} // notify parallel computation of this component is done
		}(z)
	}
	// wait for all components to finish
	for i := 0; i < numChans; i++ {
		<-done
	}

	// Reconstruct skipped parts from symmetry (X)
	for z := 0; z < size[Z]; z++ {
		for y := 0; y < size[Y]; y++ {
			for x := size[X]/2 + 1; x < size[X]; x++ {
				x2 := size[X] - x
				array[X][X][z][y][x] = array[X][X][z][y][x2]
				array[X][Y][z][y][x] = -array[X][Y][z][y][x2]
				array[X][Z][z][y][x] = -array[X][Z][z][y][x2]
				array[Y][Y][z][y][x] = array[Y][Y][z][y][x2]
				array[Y][Z][z][y][x] = array[Y][Z][z][y][x2]
				array[Z][Z][z][y][x] = array[Z][Z][z][y][x2]
			}
		}
	}

	// Reconstruct skipped parts from symmetry (Y)
	for z := 0; z < size[Z]; z++ {
		for y := size[Y]/2 + 1; y < size[Y]; y++ {
			y2 := size[Y] - y
			for x := 0; x < size[X]; x++ {
				array[X][X][z][y][x] = array[X][X][z][y2][x]
				array[X][Y][z][y][x] = -array[X][Y][z][y2][x]
				array[X][Z][z][y][x] = array[X][Z][z][y2][x]
				array[Y][Y][z][y][x] = array[Y][Y][z][y2][x]
				array[Y][Z][z][y][x] = -array[Y][Z][z][y2][x]
				array[Z][Z][z][y][x] = array[Z][Z][z][y2][x]

			}
		}
	}

	// Reconstruct skipped parts from symmetry (Z)
	for z := size[Z]/2 + 1; z < size[Z]; z++ {
		z2 := size[Z] - z
		for y := 0; y < size[Y]; y++ {
			for x := 0; x < size[X]; x++ {
				array[X][X][z][y][x] = array[X][X][z2][y][x]
				array[X][Y][z][y][x] = array[X][Y][z2][y][x]
				array[X][Z][z][y][x] = -array[X][Z][z2][y][x]
				array[Y][Y][z][y][x] = array[Y][Y][z2][y][x]
				array[Y][Z][z][y][x] = -array[Y][Z][z2][y][x]
				array[Z][Z][z][y][x] = array[Z][Z][z2][y][x]
			}
		}
	}

	array[X][X][0][0][0] = -1.0 * float32(SelfDemagNx(dx, dy, dz))
	array[Y][Y][0][0][0] = -1.0 * float32(SelfDemagNy(dx, dy, dz))
	array[Z][Z][0][0][0] = -1.0 * float32(SelfDemagNz(dx, dy, dz))
	if zero_self_demag != 0 {
		array[X][X][0][0][0] += 1.0 / 3.0
		array[Y][Y][0][0][0] += 1.0 / 3.0
		array[Z][Z][0][0][0] += 1.0 / 3.0
	}
	array[X][Y][0][0][0] = 0.0
	array[X][Z][0][0][0] = 0.0
	array[Y][Z][0][0][0] = 0.0

	// for 2D these elements are zero:
	if size[Z] == 1 {
		kernel[X][Z] = nil
		kernel[Y][Z] = nil
	}

	// make result symmetric for tools that expect it so.
	kernel[Y][X] = kernel[X][Y]
	kernel[Z][X] = kernel[X][Z]
	kernel[Z][Y] = kernel[Y][Z]

	return kernel
}
