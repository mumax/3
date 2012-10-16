package core

// Channel(s) of any dimension
type Chan interface {
	Chan() []Chan1
}

// Read channel(s) of any dimension
type RChan interface {
	RChan() []RChan1
}
