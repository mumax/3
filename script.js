// random ID number for this page, to assure proper update if open in multiple browsers
var pageID = Math.floor((Math.random()* 10000000000)+1);

// auto-update rate
var tick = 300;
var autoUpdate = true;

// show error in document (non-intrusive alert())
function showErr(err){
	if (err != ""){
		document.body.style.background = "#DDDDDD";
	}else{
		document.body.style.background = "#FFFFFF";
	}
	document.getElementById("ErrorBox").innerHTML = err;
}

// show debug message in document
function msg(err){
	document.getElementById("MsgBox").innerHTML = err;
}

// wraps document.getElementById, shows error if not found
function elementById(id){
	var elem = document.getElementById(id);
	if (elem == null){
		showErr("undefined: " + id);
		return null;
	}
	return elem;
}

// called on change of auto-update button
//function setautoupdate(){
//	autoupdate =  elementById("AutoUpdate").checked;
//}

// Id of element that has focus. We don't auto-update a focused textbox
// as this would overwrite the users input.
var hasFocus = "";
function notifyfocus(id){hasFocus = id;}
function notifyblur (id){hasFocus = "";}

function setattr_(elem, attr, value){
	if (elem[attr] == null){
		showErr("settAttr: undefined: " + elem + "[" + attr + "]");
		return;
	}
	elem[attr] = value;
}

// called by server to manipulate the DOM
function setAttr(id, attr, value){
	var elem = elementById(id);
	if (elem == null){
		showErr("undefined: " + id);
		return;
	}
	setattr_(elem, attr, value);
}

// set textbox value unless focused
function setTextbox(id, value){
	if (hasFocus != id){
		elementById(id).value = value;
	}
}

// set select value unless focused
function setSelect(id, value){
	if (hasFocus != id){
		elementById(id).value = value;
	}
}


// onreadystatechange function for update http request.
// updates the DOM with new values received from server.
function updateDOM(req){
	if (req.readyState == 4) { // DONE
		if (req.status == 200) {	
			showErr("");
			var response = JSON.parse(req.responseText);	
			for(var i=0; i<response.length; i++){
				var r = response[i];
				var func = window[r.F];
				if (func == null) {
					showErr("undefined: " + r.F);
				}else{ 
					func.apply(this, r.Args);
				}
			}
		} else {
			showErr("Disconnected");	
		}
	}
}

// updates the contents of all dynamic elements.
// periodically called via setInterval()
function update(){
		try{
			var req = new XMLHttpRequest();
			req.open("POST", document.URL, true); 
			req.timeout = tick;
			req.onreadystatechange = function(){ updateDOM(req) };
			req.setRequestHeader("Content-type","application/x-www-form-urlencoded");
			req.send("id=" + pageID);
		}catch(e){
			showErr(e); // TODO: same message as update
		}
}

function doAutoUpdate(){
	if (autoUpdate){
		update();
	}
}

setInterval(doAutoUpdate, tick);

// sends event notification to server, called on button clicks etc.
function notify(id, arg){
	try{
		var req = new XMLHttpRequest();
		req.open("PUT", document.URL, false);
		var map = {"ID": id, "Arg": arg};
		req.send(JSON.stringify(map));
	}catch(e){
		showErr(e); // TODO
	}
	update();
}

function notifyel(id, key){
	notify(id, elementById(id)[key]);
}

function notifyselect(id){
	var e = elementById(id);
	var value = e.options[e.selectedIndex].text;
	notify(id, value);
}

window.onload = update;
