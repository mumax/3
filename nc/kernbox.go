package nc

type KernelBox struct {
	Kii []chan<- Block "Kii"
	Kjj []chan<- Block "Kjj"
	Kkk []chan<- Block "Kkk"
	Kjk []chan<- Block "Kjk"
	Kik []chan<- Block "Kik"
	Kij []chan<- Block "Kij"
}

func NewKernelBox() *KernelBox {
	return new(KernelBox)
}

func (box *KernelBox) Run() {
	padded := PadSize(Size())
	acc := 4
	Debug("Initializing magnetostatic kernel")
	kern := magKernel(padded, CellSize(), Periodic(), acc)
	Debug("Magnetostatic kernel ready")

	go SendBlock(box.Kii, kern[0][0])
	go SendBlock(box.Kjj, kern[1][1])
	go SendBlock(box.Kkk, kern[2][2])
	go SendBlock(box.Kjk, kern[1][2])
	go SendBlock(box.Kik, kern[0][2])
	go SendBlock(box.Kij, kern[1][2])

	Debug("kernbox: returning")
}

// Infinitely sends the block down chan.
func SendBlock(Chan []chan<- Block, block Block) {
	for s := 0; s < NumWarp(); s++ {
		Send(Chan, block)
	}
}
