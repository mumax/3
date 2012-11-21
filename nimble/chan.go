package nimble

// Chan's are used to pass data between goroutines.
// Unlike go's built-in chan only one routine can
// (should) write to it. Also, data written to it 
// is duplicated to all readers.
type Chan interface {
	ChanN() ChanN
	MemType() MemType
}

// TODO: get rid of chan1, always arb. num comp. ?
// TODO: RChan: ReadChan
