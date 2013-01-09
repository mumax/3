package nimble

const (
	CPUMemory     MemType = 1 << 0
	GPUMemory     MemType = 1 << 1
	UnifiedMemory MemType = CPUMemory | GPUMemory
)

type MemType byte

func (m MemType) GPUAccess() bool {
	return m&GPUMemory != 0
}

func (m MemType) CPUAccess() bool {
	return m&CPUMemory != 0
}
