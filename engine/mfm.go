package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"reflect"
)

var (
	MFM        setter
	MFMLift    simpleParam
	MFMTipSize simpleParam
	mfmconv_   *cuda.MFMConvolution
)

func init() {
	MFM.init(SCALAR, &globalmesh, "MFM", "", "MFM image", SetMFM)
	MFMLift = sParam(50e-9, reinitmfmconv)
	MFMTipSize = sParam(1e-3, reinitmfmconv)
	DeclLValue("MFMLift", &MFMLift, "MFM lift height")
	DeclLValue("MFMTipSize", &MFMTipSize, "MFM tip size")
}

func SetMFM(dst *data.Slice) {

	buf := cuda.Buffer(3, Mesh())
	defer cuda.Recycle(buf)
	if mfmconv_ == nil {
		mfmconv_ = cuda.NewMFM(Mesh(), MFMLift.v, MFMTipSize.v)
	}

	mfmconv_.Exec(buf, M.Buffer(), vol(), Bsat.gpuLUT1(), regions.Gpu())
	cuda.Madd3(dst, buf.Comp(0), buf.Comp(1), buf.Comp(2), 1, 1, 1)
}

func sParam(v float64, onSet func()) simpleParam {
	return simpleParam{v: v, onSet: onSet}
}

type simpleParam struct {
	v     float64
	onSet func()
}

func (p *simpleParam) NComp() int {
	return 1
}
func (p *simpleParam) Unit() string {
	return "m"
}
func (p *simpleParam) getRegion(int) []float64 {
	return []float64{float64(p.v)}
}
func (p *simpleParam) setRegion(r int, v []float64) {
	p.v = v[0]
}

func (p *simpleParam) IsUniform() bool {
	return true
}

func (p *simpleParam) Eval() interface{} {
	return p.v
}

func (p *simpleParam) SetValue(v interface{}) {
	p.v = v.(float64)
	p.onSet()
}

func reinitmfmconv() {
	SetBusy("calculating MFM kernel")
	defer SetBusy("")
	if mfmconv_ == nil {
		mfmconv_ = cuda.NewMFM(Mesh(), MFMLift.v, MFMTipSize.v)
	} else {
		mfmconv_.Reinit(MFMLift.v, MFMTipSize.v)
	}
}

func (p *simpleParam) Type() reflect.Type {
	return reflect.TypeOf(float64(0))
}
