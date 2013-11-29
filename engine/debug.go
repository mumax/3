package engine

import (
	"fmt"
	"sync"
)

var (
	reqlock sync.Mutex
	reqs    int
)

func Req(delta int) {
	reqlock.Lock()
	defer reqlock.Unlock()
	reqs += delta
	if reqs > 3 && delta > 0 {
		fmt.Println(reqs, "pending http requests") // could ignore requests if flooded
	}
}
