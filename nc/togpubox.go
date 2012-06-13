package nc

// Copies to gpu.

type ToGpuBox struct {
	Input  <-chan []float32
	Output []chan<- GpuFloats
}

func NewToGpuBox() *ToGpuBox{
	box := new(ToGpuBox)
	Register(box)
	return box
}

func (box *ToGpuBox) Run() {

	for {
		in := Recv(box.Input)	
		buffer := GpuBuffer()
		cu.MemcpyHtoD(unsafe.Pointer(&in[0]), buffer, cu.SIZEOF_FLOAT32*WarpLen())
	}	
}
