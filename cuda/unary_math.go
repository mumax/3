package cuda

import (
	"unsafe"

	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// applyUnary is a helper that applies a pointwise async kernel
// to every component of dst and a. dst and a must have the same shape.
func applyUnary(dst, a *data.Slice, kernel func(dst, a unsafe.Pointer, N int, cfg *config)) {
	N := dst.Len()
	cfg := make1DConf(N)
	for c := 0; c < dst.NComp(); c++ {
		kernel(dst.DevPtr(c), a.DevPtr(c), N, cfg)
	}
}

// for inputs outside of a function's domain, return 0
//

// dst[i] = sin(a[i])
func QSin(dst, a *data.Slice) { applyUnary(dst, a, k_unary_sin_async) }

// dst[i] = cos(a[i])
func QCos(dst, a *data.Slice) { applyUnary(dst, a, k_unary_cos_async) }

// dst[i] = tan(a[i])
// poles at π/2 + nπ, returns 0
func QTan(dst, a *data.Slice) { applyUnary(dst, a, k_unary_tan_async) }

// dst[i] = exp(a[i])
func QExp(dst, a *data.Slice) { applyUnary(dst, a, k_unary_exp_async) }

// dst[i] = log(a[i])
func QLog(dst, a *data.Slice) { applyUnary(dst, a, k_unary_log_async) }

// dst[i] = |a[i]|
func QAbs(dst, a *data.Slice) { applyUnary(dst, a, k_unary_abs_async) }

// dst[i] = acos(a[i])
// returns 0 for a[i] outside of domain [-1, 1]
func QAcos(dst, a *data.Slice) { applyUnary(dst, a, k_unary_acos_async) }

// dst[i] = acosh(a[i])
// returns 0 for a[i] outside of domain [1, inf)
func QAcosh(dst, a *data.Slice) { applyUnary(dst, a, k_unary_acosh_async) }

// dst[i] = asin(a[i])
// returns 0 for a[i] outside of domain [-1, 1]
func QAsin(dst, a *data.Slice) { applyUnary(dst, a, k_unary_asin_async) }

// dst[i] = asinh(a[i])
func QAsinh(dst, a *data.Slice) { applyUnary(dst, a, k_unary_asinh_async) }

// dst[i] = atan(a[i])
func QAtan(dst, a *data.Slice) { applyUnary(dst, a, k_unary_atan_async) }

// dst[i] = atanh(a[i])
// returns 0 for a[i] outside of domain (-1, 1)
func QAtanh(dst, a *data.Slice) { applyUnary(dst, a, k_unary_atanh_async) }

// dst[i] = cosh(a[i])
func QCosh(dst, a *data.Slice) { applyUnary(dst, a, k_unary_cosh_async) }

// dst[i] = sinh(a[i])
func QSinh(dst, a *data.Slice) { applyUnary(dst, a, k_unary_sinh_async) }

// dst[i] = tanh(a[i])
func QTanh(dst, a *data.Slice) { applyUnary(dst, a, k_unary_tanh_async) }

// dst[i] = erf(a[i])
func QErf(dst, a *data.Slice) { applyUnary(dst, a, k_unary_erf_async) }

// dst[i] = erfc(a[i])
func QErfc(dst, a *data.Slice) { applyUnary(dst, a, k_unary_erfc_async) }

// dst[i] = tgamma(a[i])
// returns 0 for a[i] outside of domain (0, inf) with poles at non-positive integers
func QGamma(dst, a *data.Slice) { applyUnary(dst, a, k_unary_gamma_async) }

// dst[i] = 0 if a[i] < 0, 0.5 if a[i] == 0, 1 if a[i] > 0.
func QHeaviside(dst, a *data.Slice) { applyUnary(dst, a, k_unary_heaviside_async) }

// dst[i] = sin(a[i])/a[i], 1 for a[i] == 0.
func QSinc(dst, a *data.Slice) { applyUnary(dst, a, k_unary_sinc_async) }

// dst[i] = a[i]^b[i]
// pow(a,b) for negative a and b returns -pow(-a,b) for fractional exponents, and for 0^0 returns 1
func QPow(dst, src1, src2 *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src1.NComp() == nComp && src2.Len() == N && src2.NComp() == nComp)

	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_pw_pow_async(dst.DevPtr(c), src1.DevPtr(c), src2.DevPtr(c), N, cfg)
	}
}

// dst[i] = a[i] mod b[i]
func QMod(dst, src1, src2 *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src1.NComp() == nComp && src2.Len() == N && src2.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_pw_mod_async(dst.DevPtr(c), src1.DevPtr(c), src2.DevPtr(c), N, cfg)
	}
}

// dst[i] = atan2(a[i], b[i])
// returns 0 for atan2(0,0)
func QAtan2(dst, a, b *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(a.Len() == N && a.NComp() == nComp && b.Len() == N && b.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_pw_atan2_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), N, cfg)
	}
}
