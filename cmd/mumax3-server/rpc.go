package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"time"
)

func (n *Node) RPCCall(addr, method string, args ...interface{}) (interface{}, error) {
	//log.Println("call", addr, method, args)
	c := http.Client{
		Timeout: 2 * time.Second,
	}

	var buf bytes.Buffer
	for _, a := range args {
		b, err := json.Marshal(a)
		Fatal(err)
		(&buf).Write(([]byte)("/"))
		(&buf).Write(b)
	}

	URL := url.URL{
		Scheme: "http",
		Host:   addr,
		Path:   "/call/" + method + "/" + (&buf).String(),
	}
	resp, err := c.Get(URL.String())
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	//log.Println("call", addr, method, args, " -> ", string(body))

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s", body)
	}

	meth := reflect.ValueOf(n).MethodByName(method)
	nOut := meth.Type().NumOut()
	if nOut == 0 {
		return nil, nil
	}
	if nOut > 1 {
		panic("rpc supports only 1 return value")
	}

	outT := meth.Type().Out(0)
	out := reflect.New(outT).Interface()

	err = json.Unmarshal(body, out)
	if err != nil {
		//panic(err) // todo: rm
		return nil, err
	}

	return reflect.ValueOf(out).Elem().Interface(), nil
}

func (n *Node) HandleRPC(w http.ResponseWriter, r *http.Request) {
	//log.Println("got called", r.URL.Path)

	//defer func() {
	//	if err := recover(); err != nil {
	//		log.Println("RPC panic", err)
	//		http.Error(w, fmt.Sprint(err), http.StatusInternalServerError)
	//	}
	//}()

	if n.value.Kind() == 0 {
		n.value = reflect.ValueOf(n)
	}

	split := strings.Split(r.URL.Path, "/") // split = {"", "call", methodname, args...}
	name := split[2]
	args := split[3:]
	meth := n.value.MethodByName(name)
	if meth.Kind() == 0 {
		http.Error(w, "call: no such method: "+name, http.StatusBadRequest)
		return
	}

	methT := meth.Type()
	nArgs := methT.NumIn()
	argV := make([]reflect.Value, len(args))
	for i := 0; i < nArgs; i++ {
		argT := methT.In(i)
		arg := reflect.New(argT).Interface()

		err := json.Unmarshal(([]byte)(args[i]), (reflect.ValueOf(arg)).Interface())
		if err != nil {
			//panic(err) // TODO: rm
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		argV[i] = reflect.ValueOf(arg).Elem()
	}

	ret := meth.Call(argV)

	if methT.NumOut() == 0 {
		return
	}
	if methT.NumOut() > 1 {
		http.Error(w, "only 1 return value supported", http.StatusBadRequest)
	}

	retV := ret[0].Interface()
	resp, err := json.Marshal(retV)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Write(resp)
}

func (n *Node) Ping(peerInf NodeInfo) NodeInfo {
	// Somebody just called my status,
	// and him as a peer (if not yet so).
	if _, ok := n.PeerInfo(peerInf.Addr); !ok {
		n.AddPeer(peerInf)
	}
	return n.Info()
}
