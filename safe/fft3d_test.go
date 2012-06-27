package safe

import (
	"fmt"
)

func ExampleFFT3DR2C() {
	InitCuda()

	Nx, Ny, Nz := 2, 4, 8

	fft := FFT3DR2C(Nx, Ny, Nz)
	defer fft.Destroy()

	input := MakeFloat32s(fft.InputLen())
	defer input.Free()

	inputData := make([]float32, Nx*Ny*Nz)
	inputData[0*Ny*Nz] = 1
	inputData[1*Ny*Nz] = 1
	input.CopyHtoD(inputData)

	output := MakeComplex64s(fft.OutputLen())
	defer output.Free()

	fft.Exec(input, output)

	fmt.Println("input:", Reshape3DFloat32(input.Host(), Nx, Ny, Nz))
	Ox, Oy, Oz := fft.OutputSize()
	fmt.Println("output:", Reshape3DComplex64(output.Host(), Ox, Oy, Oz))

	// Output:
	// input: [[[1 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0]] [[1 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0]]]
	// output: [[[(2+0i) (+2+0i) (+2+0i) (+2-0i) (+2+0i)] [(+2+0i) (+2+0i) (+2+0i) (+2-0i) (+2+0i)] [(+2+0i) (+2+0i) (+2+0i) (+2-0i) (+2+0i)] [(+2+0i) (+2+0i) (+2+0i) (+2-0i) (+2+0i)]] [[(+0+0i) (+0+0i) (+0+0i) (+0-0i) (+0+0i)] [(+0+0i) (+0+0i) (+0+0i) (+0-0i) (+0+0i)] [(+0+0i) (+0+0i) (+0+0i) (+0-0i) (+0+0i)] [(+0+0i) (+0+0i) (+0+0i) (+0-0i) (+0+0i)]]]
}

func ExampleFFT3DC2R() {
	InitCuda()

	Nx, Ny, Nz := 2, 4, 8

	fft := FFT3DC2R(Nx, Ny, Nz)
	defer fft.Destroy()

	input := MakeComplex64s(fft.InputLen())
	defer input.Free()

	inputData := make([]complex64, fft.InputLen())
	for i := range inputData {
		inputData[i] = 2
	}
	input.CopyHtoD(inputData)

	output := MakeFloat32s(fft.OutputLen())
	defer output.Free()

	fft.Exec(input, output)

	Ix, Iy, Iz := fft.InputSize()
	fmt.Println("input:", Reshape3DComplex64(input.Host(), Ix, Iy, Iz))
	fmt.Println("output:", Reshape3DFloat32(output.Host(), Nx, Ny, Nz))

	// Output:
	// input: [[[(2+0i) (+2+0i) (+2+0i) (+2+0i) (+2+0i)] [(+2+0i) (+2+0i) (+2+0i) (+2+0i) (+2+0i)] [(+2+0i) (+2+0i) (+2+0i) (+2+0i) (+2+0i)] [(+2+0i) (+2+0i) (+2+0i) (+2+0i) (+2+0i)]] [[(+2+0i) (+2+0i) (+2+0i) (+2+0i) (+2+0i)] [(+2+0i) (+2+0i) (+2+0i) (+2+0i) (+2+0i)] [(+2+0i) (+2+0i) (+2+0i) (+2+0i) (+2+0i)] [(+2+0i) (+2+0i) (+2+0i) (+2+0i) (+2+0i)]]]
	// output: [[[128 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0]] [[0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0]]]
}

func ExampleFFT3D() {
	InitCuda()

	Nx, Ny, Nz := 2, 4, 8

	forward := FFT3DR2C(Nx, Ny, Nz)
	defer forward.Destroy()

	input := MakeFloat32s(forward.InputLen())
	defer input.Free()

	inputData := make([]float32, forward.InputLen())
	inputData[5] = 1
	input.CopyHtoD(inputData)

	output := MakeComplex64s(forward.OutputLen())
	defer output.Free()

	forward.Exec(input, output)

	backward := FFT3DC2R(Nx, Ny, Nz)
	backward.Exec(output, input)

	fmt.Println("input:", Reshape3DFloat32(inputData, Nx, Ny, Nz))
	fmt.Println("forward+inverse:", Reshape3DFloat32(input.Host(), Nx, Ny, Nz))

	// Output:
	// input: [[[0 0 0 0 0 1 0 0] [0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0]] [[0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0]]]
	// forward+inverse: [[[0 0 0 0 0 64 0 0] [0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0]] [[0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0]]]
}
