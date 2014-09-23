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
	"GiveJob": GiveJob,
}

func HandleRPC(w http.ResponseWriter, r *http.Request) {

	var ret string

	defer func() {
		log.Println(" < call  ", r.Host, r.URL.Path, "->", ret)
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

//// http Handler for incoming RPC calls.
//func (n *Node) HandleRPC(w http.ResponseWriter, r *http.Request) {
//
//	name := "?"
//	var argV []reflect.Value
//	var retV interface{}
//
//	// log debug info
//	if *flag_log {
//		defer func() {
//			var buf bytes.Buffer
//			fmt.Fprint(&buf, " < called ", name, " [")
//			for _, a := range argV {
//				fmt.Fprint(&buf, a.Interface(), " ")
//			}
//			fmt.Fprint(&buf, "] ->", retV)
//			log.Printf("%s\n", (&buf).Bytes())
//		}()
//	}
//
//	defer r.Body.Close()
//	// crash protection
//	//	defer func() {
//	//		if err := recover(); err != nil {
//	//			log.Println("RPC panic", err)
//	//			http.Error(w, fmt.Sprint(err), http.StatusInternalServerError)
//	//		}
//	//	}()
//
//	// initialize node's own reflect value
//	if n.value.Kind() == 0 {
//		n.value = reflect.ValueOf(n)
//	}
//
//	// get method name from url
//	split := strings.Split(r.URL.Path, "/")
//	name = split[2]
//	meth := n.value.MethodByName(name)
//	if meth.Kind() == 0 {
//		log.Println("handle rpc error: no such method:", name)
//		http.Error(w, "call: no such method: "+name, http.StatusBadRequest)
//		return
//	}
//
//	// prepare pointers-to-arguments with correct type
//	methT := meth.Type()
//	nArgs := methT.NumIn()
//	args := make([]interface{}, nArgs)
//	for i := 0; i < nArgs; i++ {
//		argT := methT.In(i)
//		args[i] = reflect.New(argT).Interface()
//	}
//
//	// JSON-decode argument values from request body
//	decoder := json.NewDecoder(r.Body)
//	for i := range args {
//		err := decoder.Decode(&args[i])
//		if err != nil {
//			log.Println("handle rpc error: ", err)
//			//panic(err) // TODO: rm
//			http.Error(w, err.Error(), http.StatusBadRequest)
//			return
//		}
//	}
//
//	// turn arguments into reflect.Values and call method
//	argV = make([]reflect.Value, len(args))
//	for i := range argV {
//		argV[i] = reflect.ValueOf(args[i]).Elem()
//	}
//	ret := meth.Call(argV)
//
//	//
//	if methT.NumOut() == 0 {
//		return
//	}
//	if methT.NumOut() > 1 {
//		http.Error(w, "only 1 return value supported", http.StatusBadRequest)
//	}
//
//	retV = ret[0].Interface()
//	resp, err := json.Marshal(retV)
//	if err != nil {
//		http.Error(w, err.Error(), http.StatusInternalServerError)
//		return
//	}
//	w.Write(resp)
//}

// re-usable http client for making RPC calls
var httpClient = http.Client{Timeout: 2 * time.Second}

// make RPC call to method on node with given address.
func RPCCall(addr, method, arg string) (ret string, err error) {

	defer log.Println(" > call  ", addr, method, arg, "->", ret, err)

	//TODO: escape args?
	resp, err := httpClient.Get("http://" + addr + "/do/" + method + "/" + arg)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("http status %v", resp.Status)
	}

	if b, err := ioutil.ReadAll(resp.Body); err != nil {
		return "", err
	} else {
		return string(b), nil
	}

}
