package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	MagFFTsquared   = NewVectorField("mfft_squared", "Spatial FFT[m] squared", "", SetMagFFTsquared)
	MagFFTreal      = NewVectorField("mfft_real", "Real part of spatial FFT[m]", "", SetMagFFTreal)
	MagFFTimaginary = NewVectorField("mfft_imaginary", "Imaginary part of spatial FFT[m]", "", SetMagFFTimaginary)
)

func SetMagFFTsquared(dst *data.Slice) {
	mfull := cuda.Buffer(3, M.Buffer().Size())
	defer cuda.Recycle(mfull)
	M_full.EvalTo(mfull)

	// Temporary buffers to store the real and imaginary parts of fft[m]
	fftReal := cuda.Buffer(3, mfull.Size())
	fftImag := cuda.Buffer(3, mfull.Size())
	defer cuda.Recycle(fftReal)
	defer cuda.Recycle(fftImag)

	// Compute the real and imaginary parts of fft[m] for each component
	cuda.SingleFFT(fftReal, fftImag, mfull)

	// Compute |fft[m]|^2 = Re[fft[m]]^2 + Im[fft[m]]^2 (component-wise)
	cuda.Mul(fftReal, fftReal, fftReal) // fftReal <- fftReal*fftReal
	cuda.Mul(fftImag, fftImag, fftImag) // fftImag <- fftImag*fftImag
	cuda.Madd2(dst, fftReal, fftImag, 1, 1)
}

func SetMagFFTreal(fftReal *data.Slice) {
	mfull := cuda.Buffer(3, M.Buffer().Size())
	defer cuda.Recycle(mfull)
	M_full.EvalTo(mfull)

	// Temporary buffer to store the imaginary part of fft[m]
	fftImag := cuda.Buffer(3, mfull.Size())
	defer cuda.Recycle(fftImag)

	// Compute the real and imaginary parts of fft[m] for each component
	cuda.SingleFFT(fftReal, fftImag, mfull)
}

func SetMagFFTimaginary(fftImag *data.Slice) {
	mfull := cuda.Buffer(3, M.Buffer().Size())
	defer cuda.Recycle(mfull)
	M_full.EvalTo(mfull)

	// Temporary buffer to store the real part of fft[m]
	fftReal := cuda.Buffer(3, mfull.Size())
	defer cuda.Recycle(fftReal)

	// Compute the real and imaginary parts of fft[m] for each component
	cuda.SingleFFT(fftReal, fftImag, mfull)
}
