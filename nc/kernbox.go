package nc

type KernelBox struct{}

func (box *KernelBox) Run() {
	padded := PadSize(Size())
	acc := 4
	//kern := magKernel(padded, CellSize(), Periodic(), acc)
}
