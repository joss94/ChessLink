'use strict';

/* globals MediaRecorder */
// Spec is at http://dvcs.w3.org/hg/dap/raw-file/tip/media-stream-capture/RecordingProposal.html
let constraints = {
    audio: false,
    video: {
        width: {
            min: 640,
            ideal: 640,
            max: 1920
        },
        height: {
            min: 480,
            ideal: 480,
            max: 1080
        },
        framerate: 15,
        facingMode: 'environment',
    }
};

const startBtn = document.querySelector('button#start');
const stopBtn = document.querySelector('button#stop');
const liveVideoElement = document.querySelector('#live');
const messageElement = document.querySelector('#message');

liveVideoElement.controls = false;
messageElement.style.display = "none";

let mediaRecorder = null;
let chunks = [];
let localStream = null;
let containerType = "video/mp4"; //defaults to webm but we switch to mp4 on Safari 14.0.2+

initializeApp();
startStream();

function initializeApp() {
    startBtn.style.display = "block";
    stopBtn.style.display = "none";
    messageElement.style.display = "none";
}

function startStream() {
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


function startRecording() {
    if (localStream == null) {
        alert('Could not get local stream from mic/camera');
    } else {

        /* use the stream */
        console.log('Start recording...');
        if (typeof MediaRecorder.isTypeSupported == 'function') {
            /*
            	MediaRecorder.isTypeSupported is a function announced in https://developers.google.com/web/updates/2016/01/mediarecorder
                and later introduced in the MediaRecorder API spec http://www.w3.org/TR/mediastream-recording/
            */
            let options;
            if (MediaRecorder.isTypeSupported('video/webm;codecs=vp9')) {
                options = {
                    mimeType: 'video/webm;codecs=vp9',
                    videoBitsPerSecond: 5000000,
                };
            } else if (MediaRecorder.isTypeSupported('video/webm;codecs=h264')) {
                options = {
                    mimeType: 'video/webm;codecs=h264',
                    videoBitsPerSecond: 5000000,
                };
            } else if (MediaRecorder.isTypeSupported('video/webm')) {
                options = {
                    mimeType: 'video/webm',
                    videoBitsPerSecond: 5000000,
                };
            } else if (MediaRecorder.isTypeSupported('video/mp4')) {
                //Safari 14.0.2 has an EXPERIMENTAL version of MediaRecorder enabled by default
                containerType = "video/mp4";
                options = {
                    mimeType: 'video/mp4',
                    videoBitsPerSecond: 5000000,
                };
            }
            console.log('Using ' + options.mimeType);
            mediaRecorder = new MediaRecorder(localStream, options);
        } else {
            console.log('isTypeSupported is not supported, using default codecs for browser');
            mediaRecorder = new MediaRecorder(localStream);
        }

        mediaRecorder.ondataavailable = function(e) {
            console.log('mediaRecorder.ondataavailable, e.data.size=' + e.data.size);

            chunks = [];
            if (e.data && e.data.size > 0) {
                chunks.push(e.data);
            }

            let recording = new Blob(chunks, {
                type: mediaRecorder.mimeType
            });

            // *** Upload recording video to HTTP server ***
            let formData = new FormData();
            formData.append("video", recording);
            let currentURL = window.location.protocol + "//" + window.location.hostname +":" + window.location.port;
            fetch(currentURL+'/upload', {
                    method: 'POST',
                    body: formData,
                })
                .then(response => response.json()) // Parse the data as JSON.
                .then(data => {
                    if (!data) {
                        showPopupMessage("Streaming error, cannot reach server", 5000)
                        stopRecording()
                    }
                })
                .catch(error => {
                    console.error('There has been an error with your fetch operation:', error);
                    showPopupMessage("Streaming error, cannot reach server", 5000)
                    stopRecording()
                });
        };

        mediaRecorder.onerror = function(e) {
            console.log('mediaRecorder.onerror: ' + e);
            startBtn.style.display = "block";
            stopBtn.style.display = "none";
        };

        mediaRecorder.onstart = function() {
            console.log('mediaRecorder.onstart, mediaRecorder.state = ' + mediaRecorder.state);
            startBtn.style.display = "none";
            stopBtn.style.display = "block";
        };

        mediaRecorder.onstop = function() {
            console.log('mediaRecorder.onstop, mediaRecorder.state = ' + mediaRecorder.state);
            startBtn.style.display = "block";
            stopBtn.style.display = "none";
        };

        mediaRecorder.start(1000);
        localStream.getTracks().forEach(function(track) {
            console.log(track.getSettings());
        })
    }
}

function stopRecording() {
    if (mediaRecorder != null) {
        mediaRecorder.stop();
    }
}

function showPopupMessage(message, timeout) {
    messageElement.style.display = 'block';
    messageElement.innerText = message;
    setTimeout(function(){
        messageElement.style.display = 'none'; // hide the message after 5 seconds
    }, timeout);
}

function onStartClicked() {
    startRecording()
}

function onStopClicked() {
    stopRecording()
}

navigator.mediaDevices.ondevicechange = function(event) {
    console.log("mediaDevices.ondevicechange");
}
