// +build ignore
package main

import (
	"flag"
	"fmt"
	"net/http"
	"os/exec"
	"time"
)

var (
	flag_http = flag.String("http", ":8080", "http port")
)

func echoHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, r.URL.Path[len("/echo/"):])
}

func sleepHandler(w http.ResponseWriter, r *http.Request) {
	d, err := time.ParseDuration(r.URL.Path[len("/sleep/"):])
	if err == nil {
		fmt.Fprintln(w, "sleeping for", d)
		time.Sleep(d)
		fmt.Fprintln(w, "awake again")
	} else {
		fmt.Fprint(w, err)
	}
}

func topHandler(w http.ResponseWriter, r *http.Request) {
	out, err := exec.Command("top", "-b", "-n", "1").CombinedOutput()
	if err == nil {
		w.Write(out)
	} else {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

var note = "You can leave a note here."

func noteHandler(w http.ResponseWriter, r *http.Request) {
	//title := r.URL.Path[len("/note"):]
	fmt.Fprintf(w, "<html>"+
		"<form action=\"/save/\" method=\"POST\">"+
		"<textarea name=\"body\">%s</textarea><br>"+
		"<input type=\"submit\" value=\"Save\">"+
		"</form>"+
		"</html>",
		note)
}

func saveHandler(w http.ResponseWriter, r *http.Request) {
	//title := r.URL.Path[lenPath:]
	note = r.FormValue("body")
	http.Redirect(w, r, "/note", http.StatusFound)
}

func rootHandler(w http.ResponseWriter, r *http.Request) {
	// TODO: handle trailing url parts like /notapageblabla
	fmt.Fprint(w, "nimble-cube web interface")
}

func main() {
	flag.Parse()
	//http.HandleFunc("/", rootHandler)
	http.HandleFunc("/echo/", echoHandler)
	http.HandleFunc("/top", topHandler)
	http.HandleFunc("/sleep/", sleepHandler)
	http.HandleFunc("/note", noteHandler)
	http.HandleFunc("/save/", saveHandler)
	http.Handle("/files/", http.StripPrefix("/files/", http.FileServer(http.Dir("/"))))
	http.ListenAndServe(*flag_http, nil)
}
