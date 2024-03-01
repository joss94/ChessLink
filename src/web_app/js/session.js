'use strict';

const FPS = 5;

/* globals MediaRecorder */
// Spec is at http://dvcs.w3.org/hg/dap/raw-file/tip/media-stream-capture/RecordingProposal.html
let constraints = {
    audio: false,
    video: {
        width: {
            min: 640,
            ideal: 1080,
            max: 1080
        },
        height: {
            min: 480,
            ideal: 1080,
            max: 1080
        },
        framerate: FPS,
        facingMode: 'environment',
    }
};

const stopBtn = document.querySelector('button#stop');
const canvas = document.getElementById('canvas');
const canvasOutput = document.getElementById('canvasOutput');
const messageElement = document.querySelector('#message');
const motionStatusElement = document.querySelector('#motion_status');
const boardStatusElement = document.querySelector('#board_status');

const STILL_TIME = 5;

const liveVideoElement = document.querySelector('#live-video');
liveVideoElement.controls = false;

const liveImgElement = document.querySelector('#live-img');

let intervalId = null
let mediaRecorder = null;
let localStream = null;
let session_id = "";
let cap = null;
let frame = null;
let last_frame = null;
let board_mask = null;
let board_rect = null;
let still = 0
let board_found = false
let running = false

let last_board_det_time = null;

let processing_status = null;

const pause = time => new Promise(resolve => setTimeout(resolve, time));

initializeApp();

function onOpenCvReady() {
    console.log("OpenCV was loaded !")
    startStream();
}

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
    else if (streamType == "URL") {
        const url = urlParams.get('stream_url')
        startURLStream(url)
    }
    else if (streamType == "File") {
        startFileStream()
    }
}

function startCameraStream() {

    liveImgElement.style.display = "none";
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

            runPeriodically(CameraCallback, 1000/FPS)

        }).catch(function(err) {
            console.log('navigator.getUserMedia error: ' + err);
        });
}

async function CameraCallback() {

    let width = liveVideoElement.videoWidth;
    let height = liveVideoElement.videoHeight;


    if (width != 0 && height != 0) {

        liveVideoElement.width = liveVideoElement.videoWidth;
        liveVideoElement.height = liveVideoElement.videoHeight;

        if (cap == null) {
            cap = new cv.VideoCapture(liveVideoElement);
        }

        if (frame == null) {
            frame = new cv.Mat(height, width, cv.CV_8UC4);
        }

        cap.read(frame);
        detectMotion()
    }
}

function startURLStream(url) {
    liveVideoElement.style.display = "none";
    liveImgElement.src = url;
    runPeriodically(URLCallback, 1000/FPS)
}

async function URLCallback() {
    let width = liveImgElement.width;
    let height = liveImgElement.height;

    if (width != 0 && height != 0) {
        if (frame != null) {
            frame.delete()
            frame = null
        }
        frame = cv.imread(liveImgElement)
        detectMotion()
    }
}

async function runPeriodically(callback, time) {
    running = true
    while (true) {
      callback()
      await pause(time)
    }
}

function startFileStream() {
    console.log("Not yet available")
}

function detectMotion() {

    // Detect if something is moving in the board area
    if (board_mask != null && last_frame != null) {
        let move_count = compute_motion_mask(frame, last_frame, board_mask, board_rect)

        if (move_count == 0) {
            still = still + 1
            motionStatusElement.innerText = "Still";
        } else {
            still = 0
            motionStatusElement.innerText = "Moving";
        }

        cv.imshow(canvasOutput, frame.roi(board_rect))
    }

    // If the image has been still for a long time OR
    // If the board was not detected yet
    // If 30 sec have passed since the last board detection
    //     --> Then request a frame parsing from the server
    if (still == STILL_TIME || last_board_det_time == null || (Date.now() - last_board_det_time) > 30000) {
        if (processing_status != "pending") {
            processing_status = "pending"
            processFrame()
        }
    }

    // Replace last frame by current frame for next iteration
    if (last_frame != null) {
        last_frame.delete();
    }
    last_frame = frame.clone();
}

function processFrame() {

    console.log("Processing frame")
    let formData = new FormData();

    let frame_base64 = img_to_b64(frame)

    formData.append("frame", frame_base64);
    formData.append("session_id", session_id);
    fetch(getURL() + '/process_frame', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json()) // Parse the data as JSON.
        .then(json_rsp => {
            processing_status = "done"
            console.log("Done!")
            //console.log(json_rsp)
            if (!json_rsp) {
                throw new Error('Cannot reach server');
            }

            if (json_rsp.status != "ok") {
                throw new Error('Server returned an invalid status: ' + json_rsp.status);
            }

            let game = json_rsp.data.game;
            boardStatusElement.innerText = game.status;

            if (game.status == "OK" && game.board.length == 4){
                last_board_det_time = Date.now()
                b64_to_image(game.board_mask).then(image => {
                    board_mask = image;
                    cv.cvtColor(board_mask, board_mask, cv.COLOR_RGBA2GRAY);
                    cv.threshold(board_mask, board_mask, 125, 255, cv.THRESH_BINARY);
                    board_rect = cv.boundingRect(board_mask)
                    console.log(board_rect)
                })
            }
        })
        .catch(error => {
            console.error('There has been an error with your fetch operation:', error);
            showPopupMessage("Error processing frame", 5000);
        });
}

function img_to_b64(img) {
    cv.imshow(canvas, img);
    return canvas.toDataURL("image/jpeg", 1.0);
}

function b64_to_image(base64) {
    return new Promise((resolve, reject) => {
        let full_b64 = "data:image/png;base64," + base64;
        const img = new Image();
        img.onload = () => resolve(cv.imread(img));
        img.onerror = reject;
        img.src = full_b64;
        setTimeout(function() {
            reject;
        }, 2000)
    });
}

function compute_motion_mask(frame1, frame2, mask, roi) {

    let subframe1 = frame1.roi(roi)
    let subframe2 = frame2.roi(roi)
    let submask = mask.roi(roi)

    let img1 = new cv.Mat(subframe1.rows, subframe1.cols, cv.CV_8UC1);
    let img2 = new cv.Mat(subframe2.rows, subframe2.cols, cv.CV_8UC1);

    cv.cvtColor(subframe1, img1, cv.COLOR_RGBA2GRAY);
    cv.cvtColor(subframe2, img2, cv.COLOR_RGBA2GRAY);

    // compute grayscale blurred image difference
    let frame_diff = new cv.Mat(frame2.rows, frame2.cols, cv.CV_8UC1);
    cv.absdiff(img2, img1, frame_diff);

    cv.medianBlur(frame_diff, frame_diff, 3);

    if (mask != null) {
        cv.multiply(frame_diff, submask, frame_diff, 1/255);
    }

    let motion_mask = new cv.Mat(frame_diff.rows, frame_diff.cols, cv.CV_8UC1);
    cv.threshold(frame_diff, motion_mask, 40, 255, cv.THRESH_BINARY);

    cv.medianBlur(motion_mask, motion_mask, 3)

    // morphological operations
    let ksize = new cv.Size(5, 5);
    let kernel = cv.getStructuringElement(cv.MORPH_RECT, ksize)
    cv.morphologyEx(motion_mask, motion_mask, cv.MORPH_CLOSE, kernel)

    let move_count = 0 //cv.sumElems(motion_mask) / 255//np.sum(motion_mask) / 255
    for (let r = 0; r < motion_mask.rows; r++) {
        for (let c = 0; c < motion_mask.cols; c++) {
            move_count += motion_mask.ucharAt(r, c);
        }
    }

    subframe1.delete();
    subframe2.delete();
    img1.delete();
    img2.delete();
    frame_diff.delete();
    motion_mask.delete();

    return move_count / 255;
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

    //clearInterval(intervalId);
    running = false

    let formData = new FormData();
    fetch(getURL() + '/stop_session', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json()) // Parse the data as JSON.
        .then(json_rsp => {
            //console.log(json_rsp)
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
