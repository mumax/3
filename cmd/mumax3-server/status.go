package main

// Serves human-readable status information over http.

import (
	"html/template"
	"net/http"
)

var (
	templ = template.Must(template.New("status").Parse(templText))
)

func (n *Node) HandleStatus(w http.ResponseWriter, r *http.Request) {
	n.lock()
	defer n.unlock()
	err := templ.Execute(w, node)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func (n *Node) IPRange() string {
	return *flag_scan + ": " + *flag_ports
}

const templText = `

{{define "Job"}}
<tr>
<td class={{.Status.String}}> [{{.Status.String}}] </td>
		<td> [<a href="{{.URL}}">{{.URL}}</a>] </td>
		<td> [{{with .Status}}<a href="http://{{$.Node}}">{{$.Node}}</a>/{{$.GPU}}{{end}}] </td>
		<td> [{{with .Status}}<a href="{{$.OutputURL}}">out</a>{{end}}] </td>
		<td> [{{with .IsRunning}}<a href="http://{{$.NodeName}}:{{$.GUIPort}}">gui</a>{{end}}] </td>
		<td> [{{with .Status}}{{$.Runtime}}{{end}}] </td>
	</tr>
{{end}}

<html>

<head>
	<style>
		body{font-family:monospace; margin-left:5%; margin-top:1em}
		p{margin-left: 2em}
		h3{margin-left: 2em}
		a{text-decoration: none; color:#0000AA}
		a:hover{text-decoration: underline}
		.FAILED{color:red; font-weight:bold}
		.RUNNING{font-weight: bold; color:blue}
		.QUEUED{color:black}
		.FINISHED{color: grey}
	</style>
	<meta http-equiv="refresh" content="60">
</head>

<body>

<h1>{{.Addr}}</h1>

Uptime: {{.Uptime}} <br/>

<h2>Compute service</h2><p>

<b>mumax:</b>
{{with .MumaxVersion}}
	{{.}}
{{else}}
	not available<br/>
{{end}}
<br/>

{{with .GPUs}}
	{{range $i, $v := .}}
		<b>GPU{{$i}}</b>: {{$v.Info}}<br/>
	{{end}}
{{else}}
	No GPUs available<br/>
{{end}}
</p>


<h3>Running jobs</h3><p>
	<table>
		{{range .RunningHere}}
			{{template "Job" .}}
		{{end}}
	</table>
</p>


<h2>Job scanner</h2><p>
<b>Last scan:</b> {{.LastJobScanTime}}: {{.LastJobScanFiles}} files.
<a href="http://{{.Addr}}/call/ReScan">Click to rescan</a>
</p>


<h2>Queue service</h2><p>

<table>
	{{range $k,$v := .Users}} <tr>
		<td> <a href="#{{$k}}">{{$k}}</a> </td>
		<td> <a href="#{{$k}}_running" class=RUNNING> [{{len $v.Running}} running]</a> </td>
		<td> <b>share:</b> [{{printf "%.3f" $v.UsedShare}}h/{{$v.Share}}] </td>
	</tr>{{end}}
</table>

{{range $k,$v := .Users}}
	<a id="{{$k}}"></a><h3>{{$k}}</h3><p>
	<b>Share:</b> {{printf "%.3f" .UsedShare}}h/{{.Share}} <br/>
	<b>Jobs:</b>
		<a href="#{{$k}}_finished" class=FINISHED>[{{len .Finished}} finished]</a> 
		<a href="#{{$k}}_running"  class=RUNNING> [{{len .Running}}  running]</a>
		<a href="#{{$k}}_queued"   class=QUEUED>  [{{len .Queue}}    queued]</a> 
	<br/>
	<br/>
	<a id="{{$k}}_finished"></a>  <table> {{range $v.Finished}} {{template "Job" .}} {{end}} </table>
	<a id="{{$k}}_running" ></a>  <table> {{range $v.Running}}  {{template "Job" .}} {{end}} </table>
	<a id="{{$k}}_queued"  ></a>  <table> {{range $v.Queue}}    {{template "Job" .}} {{end}} </table>
	</p>
{{end}}
</p>

<h2>HTTPFS service</h2><p>
<b>Storage root:</b> <a href="http://{{.Addr}}/fs/">{{.RootDir}}</a>
<br/><b>Open Files:</b><br/>
{{range .FSServer.LsOF}}
	{{.}}<br/>
{{end}}
</p>


<h2>Port scanner service</h2>
Peers in IP:port range {{.IPRange}}:<br/>
{{range .Peers}}
	<a href="http://{{.Addr}}">{{.Addr}}</a><br/>
{{end}}

</body>
</html>
`
