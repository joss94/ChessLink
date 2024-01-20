'use strict';

/* globals MediaRecorder */
// Spec is at http://dvcs.w3.org/hg/dap/raw-file/tip/media-stream-capture/RecordingProposal.html
let constraints = {
    audio: false,
    video: {
        width: {
            min: 640,
            ideal: 1920,
            max: 1920
        },
        height: {
            min: 480,
            ideal: 1080,
            max: 1080
        },
        framerate: 15,
        facingMode: 'environment',
    }
};

const stopBtn = document.querySelector('button#stop');
const liveVideoElement = document.querySelector('#live');
const canvas = document.getElementById('canvas');
const messageElement = document.querySelector('#message');
const infoElement = document.querySelector('#info');

liveVideoElement.controls = false;

let mediaRecorder = null;
let localStream = null;
let session_id = ""

initializeApp();
startStream();

function getURL() {
    return window.location.protocol + "//" + window.location.hostname +":" + window.location.port;
}

function initializeApp() {
    messageElement.style.display = "none";
    canvas.style.display = "none";
}

function startStream() {

    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);
    const streamType = urlParams.get('stream')
    session_id = urlParams.get('id')

    if (streamType == "Camera") {
        startCameraStream()
    }
    else if (streamType == "RTSP") {
        startRTSPStream()
    }
    else if (streamType == "File") {
        startFileStream()
    }
}

function startCameraStream() {

    if (!navigator.mediaDevices.getUserMedia) {
        alert('navigator.mediaDevices.getUserMedia not supported on your browser, use the latest version of Firefox or Chrome');
        return;
    }

    if (window.MediaRecorder == undefined) {
        alert('MediaRecorder not supported on your browser, use the latest version of Firefox or Chrome');
        return;
    }

    navigator.mediaDevices.getUserMedia(constraints)
        .then(function(stream) {
            localStream = stream;

            localStream.getTracks().forEach(function(track) {
                if (track.kind == "video") {
                    track.onended = function(event) {
                        console.log("video track.onended Video track.readyState=" + track.readyState + ", track.muted=" + track.muted);
                    }
                }
            });

            liveVideoElement.srcObject = localStream;
            liveVideoElement.play();

        }).catch(function(err) {
            console.log('navigator.getUserMedia error: ' + err);
        });
}

function startRTSPStream() {
    console.log("Not yet available")

    /*
    let formData = new FormData();
    formData.append("url", "rtsp://tapoadmin:tapo2baby@192.168.1.102:554/stream1");
    fetch(getURL() + '/set_rtsp_url', {
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
                let game = json_rsp.data.game;
                infoElement.innerText = game.status;
            }
        })
        .catch(error => {
            console.error('There has been an error with your fetch operation:', error);
            showPopupMessage("Streaming error, cannot reach server", 5000);
        });
        */
}

function startFileStream() {
    console.log("Not yet available")
}

function processFrame() {

    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);
    const streamType = urlParams.get('stream');

    if (streamType == "Camera"){

        extractFrame(function(frame_data) {
            console.log(frame_data)

            let formData = new FormData();
            formData.append("frame", frame_data);
            formData.append("session_id", session_id);
            fetch(getURL() + '/process_frame', {
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
                        let game = json_rsp.data.game;
                        infoElement.innerText = game.status;
                    }
                })
                .catch(error => {
                    console.error('There has been an error with your fetch operation:', error);
                    showPopupMessage("Streaming error, cannot reach server", 5000);
                });
        });
    }
    else if (streamType == "RTSP") {
        let formData = new FormData();
        formData.append("session_id", session_id);
        fetch(getURL() + '/process_frame_rtsp', {
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
                    let game = json_rsp.data.game;
                    infoElement.innerText = game.status;
                }
            })
            .catch(error => {
                console.error('There has been an error with your fetch operation:', error);
                showPopupMessage("Streaming error, cannot reach server", 5000);
            });
    }
}

function extractFrame(callback) {
    const context = canvas.getContext("2d");
    let width = liveVideoElement.videoWidth;
    let height = liveVideoElement.videoHeight;
    canvas.height = height;
    canvas.width = width;
    context.drawImage(liveVideoElement, 0, 0, width, height);
    canvas.toBlob(function(blob) {
        callback(blob);
    }, 'image/png');
  }

function showPopupMessage(message, timeout) {
    messageElement.style.display = 'block';
    messageElement.innerText = message;
    setTimeout(function(){
        messageElement.style.display = 'none'; // hide the message after 5 seconds
    }, timeout);
    console.log(message)
}

function onProcessClicked() {
    processFrame()
}

function onStopClicked() {
    let formData = new FormData();
    fetch(getURL() + '/stop_session', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json()) // Parse the data as JSON.
        .then(json_rsp => {
            console.log(json_rsp)
            if (!json_rsp) {
                showPopupMessage("Streaming error, cannot reach server", 5000)
            }

            else if (json_rsp.status != "ok") {
                showPopupMessage("Server error", 5000)
            }

            else {
                console.log("Session stopped")
                window.location.href = getURL()
            }
        })
        .catch(error => {
            console.error('There has been an error with your fetch operation:', error);
            showPopupMessage("Streaming error, cannot reach server", 5000)
        });

}
