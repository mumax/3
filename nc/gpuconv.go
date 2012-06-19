package nc

type GpuConvBox struct{
	M <-chan GpuBlock "M"
	B chan<- GpuBlock "B"
	Kernel [3][3] <-chan GpuBlock
	fftBuf [3]GpuBlock
}

func(box*GpuConvBox)Run(){
	
	size := Size()

	padded := [3]int{
		size[0] * 2,
		size[1] * 2,
		size[2] * 2}

	fftSize := [3]int{
		padded[0],
		padded[1],
		padded[2]+2}

	box.fftBuf = [3]GpuBlock{MakeGpuBlock(fftSize), MakeGpuBlock(fftSize), MakeGpuBlock(fftSize)}

	for{
		for s:=0; s<NumWarp(); s++{
			for c:=0; c<3; c++{

			//m := Recv(box.M[c])
			//copyPad(

			}
		}
	}
}


func copyPad(dst GpuBlock, src GpuBlock, slice int){
	dstptr := dst.Pointer()
	D0 := dst.Size()[0]
}
