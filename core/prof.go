package core

// Profiler

var tags = make(map[string]int)

func profRegister(tag string) {
	// make sure tag is not yet in use
	if _, ok := tags[tag]; ok {
		Panic("prof: tag", tag, "already in use")
	}
	tags[tag] = 1
}

func profWriteNext(tag string, delta int) {

}

func profWriteDone(tag string) {

}
