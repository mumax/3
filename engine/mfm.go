package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	MFM        = NewScalarField("MFM", "arb.", "MFM image", SetMFM)
	MFMLift    inputValue
	MFMTipSize inputValue
	mfmconv_   *cuda.MFMConvolution
)

func init() {
	MFMLift = numParam(50e-9, "MFMLift", "m", reinitmfmconv)
	MFMTipSize = numParam(1e-3, "MFMDipole", "m", reinitmfmconv)
	DeclLValue("MFMLift", &MFMLift, "MFM lift height")
	DeclLValue("MFMDipole", &MFMTipSize, "Height of vertically magnetized part of MFM tip")
}

func SetMFM(dst *data.Slice) {
	buf := cuda.Buffer(3, Mesh().Size())
	defer cuda.Recycle(buf)
	if mfmconv_ == nil {
		reinitmfmconv()
	}

	msat := Msat.MSlice()
	defer msat.Recycle()

	mfmconv_.Exec(buf, M.Buffer(), geometry.Gpu(), msat)
	cuda.Madd3(dst, buf.Comp(0), buf.Comp(1), buf.Comp(2), 1, 1, 1)
}

func reinitmfmconv() {
	SetBusy(true)
	defer SetBusy(false)
	if mfmconv_ == nil {
		mfmconv_ = cuda.NewMFM(Mesh(), MFMLift.v, MFMTipSize.v, *Flag_cachedir)
	} else {
		mfmconv_.Reinit(MFMLift.v, MFMTipSize.v, *Flag_cachedir)
	}
}
