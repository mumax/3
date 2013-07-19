
// auto-refresh rate
var tick = 200;
var autorefresh = true;

// show error in document (non-intrusive alert())
function showErr(err){
	document.getElementById("ErrorBox").innerHTML = err;
}

// show debug message in document
function msg(err){
	document.getElementById("MsgBox").innerHTML = err;
}

// called on change of auto-refresh button
function setautorefresh(){
	autorefresh =  document.getElementById("AutoRefresh").checked;
}

// Id of element that has focus. We don't auto-refresh a focused textbox
// as this would overwrite the users input.
var hasFocus = "";
function notifyfocus(id){hasFocus = id;}
function notifyblur (id){hasFocus = "";}

function setAttr(id, attr, value){
	document.getElementById(id)[attr] = value;
}

// onreadystatechange function for update http request.
// refreshes the DOM with new values received from server.
function refreshDOM(req){
	if (req.readyState == 4) { // DONE
		if (req.status == 200) {	
			showErr("");
			var response = JSON.parse(req.responseText);	
			for(var i=0; i<response.length; i++){
				var r = response[i];
				window[r.F].apply(this, r.Args);
			}
		} else {
			showErr("Disconnected");	
		}
	}
}

// refreshes the contents of all dynamic elements.
// periodically called via setInterval()
function refresh(){
	if (autorefresh){
		try{
			var req = new XMLHttpRequest();
			req.open("POST", "/refresh/", true);
			req.timeout = tick;
			req.onreadystatechange = function(){ refreshDOM(req) };
			req.send(null);
		}catch(e){
			showErr(e); // TODO: same message as refresh
		}
	}
}

setInterval(refresh, tick);

// sends event notification to server, called on button clicks etc.
function notify(id, method, arg){
	try{
		var req = new XMLHttpRequest();
		req.open("POST", "/event/", false);
		var map = {"ID": id, "Method": method, "Arg": arg};
		req.send(JSON.stringify(map));
	}catch(e){
		showErr(e); // TODO
	}
	refresh();
}

function notifytextbox(id){
	notify(id, "change", document.getElementById(id).value);
}

function notifycheckbox(id){
	notify(id, "change", document.getElementById(id).checked);
}

function notifyrange(id){
	notify(id, "change", document.getElementById(id).value);
}

function notifyselect(id){
	notify(id, "change", document.getElementById(id).value);
}
