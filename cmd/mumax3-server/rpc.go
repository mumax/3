package main

import (
	"encoding/json"
	//"fmt"
	"log"
	"net/http"
	"reflect"
	"strings"
)

func (n *Node) HandleRPC(w http.ResponseWriter, r *http.Request) {
	log.Println("call", r.URL.Path)

	n.lock()
	defer n.unlock()

	//	defer func() {
	//		if err := recover(); err != nil {
	//			log.Println("RPC panic", err)
	//			http.Error(w, fmt.Sprint(err), http.StatusInternalServerError)
	//		}
	//	}()

	if n.value.Kind() == 0 {
		n.value = reflect.ValueOf(n)
	}

	split := strings.Split(r.URL.Path, "/")
	// split = {"", "call", methodname, args...}
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
		arg := reflect.Zero(argT)
		err := json.Unmarshal(([]byte)(args[i]), arg.Addr().Interface())
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		argV[i] = arg
	}
	ret := meth.Call(argV)
	retV := make([]interface{}, len(ret))
	for i := range retV {
		retV[i] = ret[i].Interface()
	}
	resp, err := json.MarshalIndent(retV, "", "\t")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Write(resp)
}

//func (r *RPC) Ping(peerInf NodeInfo, myInf *NodeInfo) error {
//	n := r.node
//	*myInf = n.Info()
//
//	// Somebody just called my status, let's probe him
//	// and possibly add as new peer.
//	if _, ok := n.PeerInfo(peerInf.Addr); !ok {
//		n.AddPeer(peerInf)
//		//ProbePeer(peerInf.Addr)
//	}
//	return nil
//}
