package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"strings"
	"time"
)

type RPCFunc func(string) string

var methods = map[string]RPCFunc{
	"AddFairShare":   AddFairShare,
	"GiveJob":        GiveJob,
	"Kill":           Kill,
	"LoadJobs":       wrap(LoadJobs),
	"LoadUserJobs":   LoadUserJobs,
	"Ping":           Ping,
	"UpdateJob":      UpdateJob,
	"Rescan":         func(string) string { go FindPeers(IPs, MinPort, MaxPort); return "" },
	"WhatsTheTime":   WhatsTheTime,
	"WakeupWatchdog": WakeupWatchdog,
	"rm":             Rm,
}

func wrap(f func()) RPCFunc {
	return func(string) string { f(); return "" }
}

func HandleRPC(w http.ResponseWriter, r *http.Request) {

	var ret string

	defer func() {
		//log.Println(" < call  ", r.Host, r.URL.Path, "->", ret)
		if err := recover(); err != nil {
			log.Println("*** RPC   panic: ", r.URL.Path, ":", err)
			http.Error(w, "Does not compute: "+r.URL.Path, http.StatusBadRequest)
		}
	}()
	request := r.URL.Path[len("/do/"):]
	slashPos := strings.Index(request, "/")
	method := request[:slashPos]
	arg := request[slashPos+1:]

	m, ok := methods[method]
	if !ok {
		log.Println("*** RPC   no such method", r.URL.Path)
		http.Error(w, "Does not compute: "+method, http.StatusBadRequest)
		return
	}
	ret = m(arg)
	fmt.Fprint(w, ret)
}

// re-usable http client for making RPC calls
var httpClient = http.Client{Timeout: 2 * time.Second}

// make RPC call to method on node with given address.
func RPCCall(addr, method, arg string) (ret string, err error) {

	//defer func() { log.Println(" > call  ", addr, method, arg, "->", ret, err) }()

	//TODO: escape args?
	resp, err := httpClient.Get("http://" + addr + "/do/" + method + "/" + arg)
	if err != nil {
		//log.Println("*** RPC  error: ", err)
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Println("*** RPC  error: ", resp.Status)
		return "", fmt.Errorf("http status %v", resp.Status)
	}

	if b, err := ioutil.ReadAll(resp.Body); err != nil {
		log.Println("*** RPC  read error: ", err)
		return "", err
	} else {
		return string(b), nil
	}

}
