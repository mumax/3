package safe

import (
	"fmt"
)

func ExampleFFT1DR2C() {
	InitCuda()

	N := 8
	batch := 1

	fft := FFT1DR2C(N, batch)
	defer fft.Destroy()

	input := MakeFloat32s(N)
	defer input.Free()
	input.CopyHtoD([]float32{1, 0, 0, 0, 0, 0, 0, 0})

	output := MakeComplex64s(fft.OutputLen())
	defer output.Free()

	fft.Exec(input, output)

	fmt.Println("input:", input.Host())
	fmt.Println("output:", output.Host())

	// Output:
	// input: [1 0 0 0 0 0 0 0]
	// output: [(1+0i) (+1+0i) (+1+0i) (+1-0i) (+1+0i)]
}

func ExampleFFT1DR2C_Inplace() {
	InitCuda()

	N := 8
	batch := 2

	fft := FFT1DR2C(N, batch)
	defer fft.Destroy()

	output := MakeComplex64s(fft.OutputLen())
	defer output.Free()

	input := output.Float().Slice(0, fft.InputLen())
	// input uses same layout as out-of-place transform
	// (CUFFT native layout)
	input.CopyHtoD([]float32{1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0})
	fmt.Println("input:", input.Host())

	fft.Exec(input, output)
	fmt.Println("output:", output.Host())

	inverse := FFT1DC2R(N, batch)
	defer inverse.Destroy()
	inverse.Exec(output, input)
	fmt.Println("input:", input.Host())

	// Output:
	// input: [1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
	// output: [(1+0i) (+1+0i) (+1+0i) (+1-0i) (+1+0i) (+1+0i) (+1+0i) (+1+0i) (+1-0i) (+1+0i)]
	// input: [8 0 0 0 0 0 0 0 8 0 0 0 0 0 0 0]
}
func ExampleFFT1DC2R() {
	InitCuda()

	N := 8
	batch := 1

	fft := FFT1DC2R(N, batch)
	defer fft.Destroy()

	input := MakeComplex64s(fft.InputLen())
	defer input.Free()
	input.CopyHtoD([]complex64{(1 + 0i), (+1 + 0i), (+1 + 0i), (+1 - 0i), (+1 + 0i)})

	output := MakeFloat32s(fft.OutputLen())
	defer output.Free()

	fft.Exec(input, output)

	fmt.Println("input:", input.Host())
	fmt.Println("output:", output.Host())

	// Output:
	// input: [(1+0i) (+1+0i) (+1+0i) (+1+0i) (+1+0i)]
	// output: [8 0 0 0 0 0 0 0]
}
