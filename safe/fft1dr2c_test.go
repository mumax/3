package safe

import (
	"fmt"
)

func ExampleFFT1DR2C() {
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
