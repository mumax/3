// +build ignore
package main

import(
	"net/http"
	"flag"
	"fmt"
	"os/exec"
)

var(
	flag_http = flag.String("http", ":8080", "http port")
)

func echoHandler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprint(w, r.URL.Path[len("/echo/"):])
}

func topHandler(w http.ResponseWriter, r *http.Request) {
	out, err := exec.Command("top", "-b", "-n", "1").CombinedOutput()
	if err == nil{
		w.Write(out)
	}else{
		fmt.Fprint(w, err)
	}
}

func rootHandler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprint(w, "nimble-cube web interface")
}

func main() {
	flag.Parse()
    http.HandleFunc("/echo/", echoHandler)
    http.HandleFunc("/top", topHandler)
    http.HandleFunc("/", rootHandler)
    http.ListenAndServe(*flag_http, nil)
}
