package core

// Protects an array for thread-safe
// concurrent reading and writing.
type RWMutex struct{
	[]RLock
}

// Locks for writing between indices 
// start (inclusive) and stop (exclusive).
func(m*RWMutex)Lock(start, stop int){

}

// Registers and returns a new lock for reading.
func(m*RWMutex)MakeRLock()*RLock{

}

// Lock for reading a RWMutex.
type RLock struct{

}

// Locks for reading between indices 
// start (inclusive) and stop (exclusive).
func(m*RLock)RLock(start, stop int){

}
