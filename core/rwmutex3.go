package core

type RWMutex3 [3]*RWMutex

func (m *RWMutex3) WriteNext(delta int) {
	for i := range m {
		m[i].WriteNext(delta)
	}
}

func (m *RWMutex3) WriteDone() {
	for i := range m {
		m[i].WriteDone()
	}
}

func (m *RWMutex3) NewReader() RMutex3 {
	return RMutex3{m[0].NewReader(), m[1].NewReader(), m[2].NewReader()}
}

type RMutex3 [3]*RMutex

func (m *RMutex3) ReadNext(delta int) {
	for i := range m {
		m[i].ReadNext(delta)
	}
}

func (m *RMutex3) ReadDone() {
	for i := range m {
		m[i].ReadDone()
	}
}
