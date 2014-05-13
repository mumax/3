package gui

// Javascript for the GUI page.

const JS = `<script type="text/javascript">

function genPageID(){ return Math.floor((Math.random()* 10000000000)+1); }

// random ID number for this page, to assure proper update if open in multiple browsers
var pageID = genPageID();

var tick = 300;         // auto-update rate
var autoUpdate = true;
var disconnects = 0;    // number of successive disconnects

// show error in document (non-intrusive alert())
function showErr(err){
	if (err != ""){ 
		disconnects++; // don't show error just yet
	}else{
		disconnects = 0;
		document.getElementById("ErrorBox").innerHTML = "";
		document.body.style.background = "#FFFFFF";
	}
	if (disconnects > 3){ // disconnected for some time, show error
		var col = (256+3)-disconnects;   // gradually fade out background color
		if(col<210){col=210;}            // not too dark
		document.body.style.background = "rgb(" + col+"," + col+"," + col+")";
		document.getElementById("ErrorBox").innerHTML = err;
	}
}

// show debug message in document
//function msg(err){
//	document.getElementById("MsgBox").innerHTML = err;
//}

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

// Id of textbox that has focus. We don't auto-update a focused textbox
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
		var el = elementById(id);
		el.value = value;
		el.style.color = "black"; // was red during edit, back up-to-date now
	}
}

// set select value unless focused
function setSelect(id, value){
	if (hasFocus != id){
		elementById(id).value = value;
	}
}

var pending = false; // only one request at a time

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
			pageID = genPageID(); // communication got lost, so refresh everything next time
		}
		pending=false;
	}
}

// updates the contents of all dynamic elements.
// periodically called via setInterval()
function update(){
		try{
			pending = true;
			var req = new XMLHttpRequest();
			req.open("POST", document.URL, true); 
			req.onreadystatechange = function(){ updateDOM(req) };
			req.setRequestHeader("Content-type","application/x-www-form-urlencoded");
			req.send("id=" + pageID);
		}catch(e){
			showErr(e); // TODO: same message as update
		}
}

function doAutoUpdate(){
	if (autoUpdate && !pending){
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
	var el = elementById(id);
	notify(id, el[key]);
}

function makered(id, event){
	var el = elementById(id);
	var key = event.keyCode;
	if (key == 13){ // return key
		hasFocus = "";  // give up focus so that value can change after hitting return
		//notifyel(id, "value"); // already done by onchange
	}else{
		hasFocus = id; // grab focus back
		el.style.color = "red"; // changed
	}
}

function notifyselect(id){
	var e = elementById(id);
	var value = e.options[e.selectedIndex].text;
	notify(id, value);
}

window.onload = update;
</script>`
