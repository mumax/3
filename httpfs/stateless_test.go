package httpfs

import (
	"net/http"
	"testing"
)

const (
	testAddr = "localhost:35371"
	testURL  = "http://" + testAddr + "/"
)

func init() {
	Handle()
	go func() {
		if err := http.ListenAndServe(testAddr, nil); err != nil {
			panic(err)
		}
	}()
}

func TestStatelessMkdir(t *testing.T) {

}

func TestStatelessReadDir(t *testing.T) {

}

func TestStatelessRemove(t *testing.T) {

}

func TestStatelessRead(t *testing.T) {

}

func TestStatelessAppend(t *testing.T) {

}
