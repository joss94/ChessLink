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
        framerate: 30,
        facingMode: 'environment',
    }
};

let liveVideoElement = document.querySelector('#live');
let messageElement = document.querySelector('#message');

liveVideoElement.controls = false;
messageElement.style.display = "none";

let mediaRecorder;
let chunks = [];
let localStream = null;
let containerType = "video/mp4"; //defaults to webm but we switch to mp4 on Safari 14.0.2+

if (!navigator.mediaDevices.getUserMedia) {
    alert('navigator.mediaDevices.getUserMedia not supported on your browser, use the latest version of Firefox or Chrome');
} else {
    if (window.MediaRecorder == undefined) {
        alert('MediaRecorder not supported on your browser, use the latest version of Firefox or Chrome');
    } else {
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
}

function onBtnRecordClicked() {
    if (localStream == null) {
        alert('Could not get local stream from mic/camera');
    } else {
        recBtn.disabled = true;
        chunks = [];

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
                    let message;
                    if (data) {
                        message = "Video Uploaded!";
                    } else {
                        message = "Video upload failed!";
                    }
                    console.log(message);
                    messageElement.style.display = 'block';
                    messageElement.innerText = message;
                    setTimeout(function(){
                        messageElement.style.display = 'none'; // hide the message after 5 seconds
                    }, 5000);
                })
                .catch(error => {
                    console.error('There has been an error with your fetch operation:', error);
                });
        };

        mediaRecorder.onerror = function(e) {
            console.log('mediaRecorder.onerror: ' + e);
        };

        mediaRecorder.onstart = function() {
            console.log('mediaRecorder.onstart, mediaRecorder.state = ' + mediaRecorder.state);
        };

        mediaRecorder.onstop = function() {
            console.log('mediaRecorder.onstop, mediaRecorder.state = ' + mediaRecorder.state);
        };

        mediaRecorder.start(1000);
        localStream.getTracks().forEach(function(track) {
            console.log(track.getSettings());
        })
    }
}

navigator.mediaDevices.ondevicechange = function(event) {
    console.log("mediaDevices.ondevicechange");
}
