package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	MFM        setter
	MFMLift    numberParam
	MFMTipSize numberParam
	mfmconv_   *cuda.MFMConvolution
)

func init() {
	MFM.init(SCALAR, &globalmesh, "MFM", "", "MFM image", SetMFM)
	MFMLift = numParam(50e-9, "m", reinitmfmconv)
	MFMTipSize = numParam(1e-3, "m", reinitmfmconv)
	DeclLValue("MFMLift", &MFMLift, "MFM lift height")
	DeclLValue("MFMTipSize", &MFMTipSize, "MFM tip size")
}

func SetMFM(dst *data.Slice) {

	buf := cuda.Buffer(3, Mesh().Size())
	defer cuda.Recycle(buf)
	if mfmconv_ == nil {
		mfmconv_ = cuda.NewMFM(Mesh(), MFMLift.v, MFMTipSize.v)
	}

	mfmconv_.Exec(buf, M.Buffer(), geometry.Gpu(), Bsat.gpuLUT1(), regions.Gpu())
	cuda.Madd3(dst, buf.Comp(0), buf.Comp(1), buf.Comp(2), 1, 1, 1)
}

func reinitmfmconv() {
	GUI.SetBusy(true)
	defer GUI.SetBusy(false)
	if mfmconv_ == nil {
		mfmconv_ = cuda.NewMFM(Mesh(), MFMLift.v, MFMTipSize.v)
	} else {
		mfmconv_.Reinit(MFMLift.v, MFMTipSize.v)
	}
}
