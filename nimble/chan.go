package nimble

// TODO: get rid of chan1, chan3, always arb. num comp.
// TODO: RChan: ReadChan
type Chan interface {
	ChanN() ChanN
}
