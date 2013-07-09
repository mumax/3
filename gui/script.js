
// auto-refresh rate
var tick = 1000;
var autorefresh = true;

// show error in document
function showErr(err){
	document.getElementById("ErrorBox").innerHTML = err;
}

// called on change of auto-refresh button
function setautorefresh(){
	autorefresh =  document.getElementById("AutoRefresh").checked;
}

// onreadystatechange function for update request
function refreshDOM(req){
	if (req.readyState == 4) { // DONE
		if (req.status == 200) {	
			showErr("");
			var response = JSON.parse(req.responseText);	
			for(var i=0; i<response.length; i++){
				var r = response[i];
				document.getElementById(r.ID)[r.Var] = r.HTML;
			}
		} else {
			showErr("Disconnected");	
		}
	}
}

// refreshes the contents of all dynamic elements,
// leaves the rest of the page alone.
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

// remote procedure call, called on button clicks etc.
function rpc(method, args){
	try{
		var req = new XMLHttpRequest();
		req.open("POST", "/rpc/", false);
		var map = {"Method": method, "Args": args};
		req.send(JSON.stringify(map));
	}catch(e){
		showErr(e); // TODO
	}
	refresh();
}

setInterval(refresh, tick);
