package core

type RMutex struct{
	rw *RWMutex
	c, d int
}
