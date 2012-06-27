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
