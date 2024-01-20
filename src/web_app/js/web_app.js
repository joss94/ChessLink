'use strict';

const startBtn = document.querySelector('button#start');
const messageElement = document.querySelector('#message');
const stream_selector = document.querySelector("select#stream_type");
const rtsp_url = document.querySelector("input#rtsp_url");

initializeApp();

function getURL() {
    return window.location.protocol + "//" + window.location.hostname +":" + window.location.port;
}

function initializeApp() {
    onStreamTypeChanged()
}

function startSession() {
    let formData = new FormData();
    fetch(getURL() + '/start_session', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json()) // Parse the data as JSON.
        .then(json_rsp => {
            console.log(json_rsp)
            if (!json_rsp) {
                showPopupMessage("Streaming error, cannot reach server", 5000);
            }

            else if (json_rsp.status != "ok") {
                showPopupMessage("Server error", 5000);
            }

            else {
                let data = json_rsp.data;
                let session_id = data["session_id"];
                console.log("Started new session: ", session_id);

                let stream_type = stream_selector.options[stream_selector.selectedIndex].text;

                url = getURL() + "/session?" + "id=" + session_id + "&stream=" + stream_type;

                window.location.href = url
                // window.location.href = getURL() + "/session";
            }

        })
        .catch(error => {
            console.error('There has been an error with your fetch operation:', error);
            showPopupMessage("Streaming error, cannot reach server", 5000);
        });
}

function showPopupMessage(message, timeout) {
    messageElement.style.display = 'block';
    messageElement.innerText = message;
    setTimeout(function(){
        messageElement.style.display = 'none'; // hide the message after 5 seconds
    }, timeout);
    console.log(message);
}

function onStartClicked() {
    startSession();
}

function onStreamTypeChanged() {
    var streamType = stream_selector.options[stream_selector.selectedIndex].text;
    console.log(streamType);
    if (streamType == "Camera") {
        rtsp_url.style.display = "none";
    }
    else if (streamType == "RTSP") {
        rtsp_url.style.display = "block";
    }
    else if (streamType == "File") {
        rtsp_url.style.display = "none";
    }
}

