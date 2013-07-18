package web

//import (
//	"code.google.com/p/mx3/engine"
//	"fmt"
//	"net/http"
//	"time"
//)
//
//// responds to "/running" with true or false
//func runningHandler(w http.ResponseWriter, r *http.Request) {
//	LastKeepalive = time.Now()
//	fmt.Fprint(w, !engine.Paused())
//}
//
//// when we last saw browser activity
//var LastKeepalive time.Time
//
//// keep session open this long after browser inactivity
//const Timeout = 60 * time.Second
