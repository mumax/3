package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	//	"log"
	"reflect"
)

var (
	MFM        setter
	MFMLift    = simpleParam(50e-9)
	MFMTipSize = simpleParam(1e-3)
	mfmconv_   *cuda.MFMConvolution
)

func init() {
	MFM.init(SCALAR, &globalmesh, "MFM", "", "MFM image", SetMFM)
	DeclLValue("MFMLift", &MFMLift, "MFM lift height")
	DeclLValue("MFMTipSize", &MFMTipSize, "MFM tip size")
}

func SetMFM(dst *data.Slice) {

	buf := cuda.Buffer(3, Mesh())
	defer cuda.Recycle(buf)
	if mfmconv_ == nil {
		mfmconv_ = cuda.NewMFM(Mesh(), float64(MFMLift), float64(MFMTipSize))
	}

	mfmconv_.Exec(buf, M.buffer, vol(), Bsat.gpuLUT1(), regions.Gpu())
	cuda.Madd3(dst, buf.Comp(0), buf.Comp(1), buf.Comp(2), 1, 1, 1)
}

type simpleParam float64

func (p *simpleParam) NComp() int {
	return 1
}
func (p *simpleParam) Unit() string {
	return "m"
}
func (p *simpleParam) getRegion(int) []float64 {
	return []float64{float64(*p)}
}
func (p *simpleParam) setRegion(r int, v []float64) {
	*p = (simpleParam)(v[0])
}

func (p *simpleParam) IsUniform() bool {
	return true
}

func (p *simpleParam) Eval() interface{} {
	return float64(*p)
}

func (p *simpleParam) SetValue(v interface{}) {
	newv := v.(float64)
	if newv != float64(*p) {
		//log.Println("oldv", *p, "newv", newv)
		*p = simpleParam(newv)
		reinitmfmconv()
	}
}

func reinitmfmconv() {
	if mfmconv_ == nil {
		mfmconv_ = cuda.NewMFM(Mesh(), float64(MFMLift), float64(MFMTipSize))
	} else {
		mfmconv_.Reinit(float64(MFMLift), float64(MFMTipSize))
	}
}

func (p *simpleParam) Type() reflect.Type {
	return reflect.TypeOf(float64(0))
}
