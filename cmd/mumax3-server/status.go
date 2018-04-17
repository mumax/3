package main

// Serves human-readable status information over http.

import (
	"html/template"
	"net/http"
	"time"
)

var (
	templ   = template.Must(template.New("status").Parse(templText))
	upSince = time.Now()
)

func HandleStatus(w http.ResponseWriter, r *http.Request) {
	RLock()
	defer RUnlock()

	if r.URL.Path != "/" {
		http.Error(w, "Does not compute", http.StatusNotFound)
		return
	}

	err := templ.Execute(w, &status{})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

type status struct{} // dummy type to define template methods on

func (*status) IPRange() string                { return *flag_scan + ": " + *flag_ports }
func (*status) Ports() string                  { return *flag_ports }
func (*status) ThisAddr() string               { return thisAddr }
func (*status) Uptime() time.Duration          { return Since(time.Now(), upSince) }
func (*status) MumaxVersion() string           { return MumaxVersion }
func (*status) GPUs() []string                 { return GPUs }
func (*status) Processes() map[string]*Process { return Processes }
func (*status) Users() map[string]*User        { return Users }
func (*status) NextUser() string               { return nextUser() }
func (*status) Peers() map[string]*Peer        { return peers }
func (*status) FS(a string) string             { return FS(a) }

const templText = `

{{define "Job"}}
<tr class={{.Status}}>
		<td class={{.Status}}> [<a class={{.Status}} href="http://{{.FS .ID}}">{{.LocalPath}}</a>] </td>
		<td class={{.Status}}> [{{with .Output}}<a href="http://{{$.FS $.Output}}">.out</a>{{end}}] </td>
		<td class={{.Status}}> [{{with .Output}}<a onclick='doEvent("rm", "{{$.ID}}")'>rm</a>{{end}}]</td>
		<td class={{.Status}}> [{{with .Host}}<a href="http://{{.}}">{{.}}</a>{{end}}] </td>
		<td class={{.Status}}> [{{with .ExitStatus}}{{if eq . "0"}} OK {{else}}<a class={{$.Status}} href="http://{{$.FS $.Output}}stdout.txt">FAIL</a>{{end}}{{end}}] </td>
		<td class={{.Status}}> [{{with .Output}}{{$.Duration}}{{end}}{{with .RequeCount}} {{.}}x re-queued{{end}}{{with .Error}} {{.}}{{end}}] </td>
</tr>
{{end}}

<html>

<head>
	<style>
		body{font-family:monospace; margin-left:5%; margin-top:1em}
		p{margin-left: 2em}
		h3{margin-left: 2em}
		a{text-decoration: none; color:#0000AA}
		a:hover{text-decoration: underline; cursor: hand;}
		.FAILED{color:red; font-weight:bold}
		.RUNNING{font-weight: bold; color:blue}
		.QUEUED{color:black}
		.FINISHED{color: grey}
		.active, .collapsible:hover {font-weight: normal; background-color: #eee; width: 50%;}
	</style>
	<meta http-equiv="refresh" content="60">
</head>

<script>
function doEvent(method, arg){
	try{
		var req = new XMLHttpRequest();
		var URL = "http://" + window.location.hostname + ":" + window.location.port + "/do/" + method + "/" + arg;
		req.open("GET", URL, false);
		req.send(null);
	}catch(e){
		alert(e);
	}
	location.reload();
}
</script>

<body>

<h1>{{.ThisAddr}}</h1>

Uptime: {{.Uptime}} <br/>

<h2>Peer nodes</h2>

	<b>scan</b> {{.IPRange}}<br/>
	<b>ports</b> {{.Ports}}<br/>
	<button onclick='doEvent("Rescan", "")'>Rescan</button> <br/>
	{{range $k,$v := .Peers}} 
		<a href="http://{{$k}}">{{$k}}</a> <br/>
	{{end}}

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
			<b>GPU{{$i}}</b>: {{$v}}<br/>
		{{end}}
	{{else}}
		No GPUs available<br/>
	{{end}}
	</p>
	
	
	<h3>Running jobs</h3><p>
		<table>
			{{range $k,$v := .Processes}}
				<tr>
					<td> [<a href="http://{{$.FS $k}}">{{$k}}</a>] </td>
					<td> [{{$v.Duration}}]</td> 
					<td> [<a href="http://{{$v.GUI}}">GUI</a>]</td> 
					<td> <button onclick='doEvent("Kill", "{{$k}}")'>kill</button> </td>
				</tr>
			{{end}}
		</table>
	</p>


<h2>Queue service</h2><p>

	<h3>Users</h3><p>
	<table>
	{{range $k,$v := .Users}} <tr>
		<td>{{$k}}</td><td>{{$v.FairShare}} GPU-seconds</td><td>{{with .HasJob}} has {{else}} no {{end}} queued jobs</td>
	</tr>{{end}}
	</table>
	<b>Next job for:</b> {{.NextUser}}
	</p>

	<h3>Jobs</h3>
		<button onclick='doEvent("LoadJobs", "")'>Reload all</button> (consider reloading just your own files). <br/>
		<button onclick='doEvent("WakeupWatchdog", "")'>Wake-up Watchdog</button> (re-queue dead simulations right now).
	{{range $k,$v := .Users}}
		<a id="{{$k}}"></a>
		<h3 class="collapsible" onclick='this.classList.toggle("active");var cont=this.nextElementSibling;cont.style.display=(cont.style.display==="none"?"block":"none");'>
		{{$k}}</h3><p>
	

		<b>Jobs</b>
		<button onclick='doEvent("LoadUserJobs", "{{$k}}")'>Reload</button> (only needed when you changed your files on disk)

		<table> {{range $v.Jobs}} {{template "Job" .}} {{end}} </table>
		</p>
	{{end}}
	</p>

</body>
</html>
`
