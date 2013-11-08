package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"log"
)

var (
	MFM        setter
	MFMLift    float64
	MFMTipSize float64
	mfmconv_   *cuda.MFMConvolution
)

func init() {
	MFM.init(SCALAR, &globalmesh, "MFM", "", "MFM image", SetMFM)
	DeclVar("MFMLift", &MFMLift, "MFM tip lift")
}

func SetMFM(dst *data.Slice) {
	buf := cuda.Buffer(3, Mesh())
	defer cuda.Recycle(buf)
	if mfmconv_ == nil {
		mfmconv_ = cuda.NewMFM(Mesh(), MFMLift, MFMTipSize)
	}
	mfmconv_.Exec(buf, M.buffer, vol(), Bsat.gpuLUT1(), regions.Gpu())
	log.Println(buf.HostCopy())
	cuda.Madd3(dst, buf.Comp(0), buf.Comp(1), buf.Comp(2), 1, 1, 1)
}
