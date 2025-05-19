hl_nav();
function hl_nav(){
	var myNav = document.getElementById("top_nav").getElementsByTagName("a");
	var myUrl = document.location.href;
	for(var i = 0; i < myNav.length; i++){
		var links = myNav[i].getAttribute("href");
		if(myUrl.indexOf(links) != -1 && links != '/'){
			myNav[i].className="active";
			break;
		}
	}
}

function setModel(){
	var obj = document.getElementById("selectList");
	var opts = obj.options;
	var tgts = document.getElementsByName("chosen_model");
	for(var i = 0; i < tgts.length; i++){
		tgts[i].value = opts[obj.selectedIndex].value;
	}
}

function changeModel(){
	var form = document.getElementById("changeModelForm");
	form.submit();
}