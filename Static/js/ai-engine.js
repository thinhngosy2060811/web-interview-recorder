async function initAI() {
    const videoElement = document.getElementById('preview');

    function getDistance(p1, p2) {
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
    }

    function getHorizontalGazeRatio(p_iris, p_left_corner, p_right_corner) {
        const dist_total = getDistance(p_left_corner, p_right_corner);
        const dist_to_left = getDistance(p_iris, p_left_corner);
        return dist_to_left / dist_total;
    }

    function getVerticalGazeRatio(p_iris, p_top_eyelid, p_bottom_eyelid) {
        const dist_total = getDistance(p_top_eyelid, p_bottom_eyelid);
        const dist_to_top = getDistance(p_iris, p_top_eyelid);
        
        if (dist_total < 0.005) return 0.5;
        
        return dist_to_top / dist_total;
    }

    function getEyeOpenRatio(p_top, p_bottom, p_inner, p_outer) {
        const height = getDistance(p_top, p_bottom);
        const width = getDistance(p_inner, p_outer);
        return height / width;
    }

    function onResults(results) {
        if (!isAiRunning) return;
        
        aiAnalysis.totalFrames++;
        let warningMsg = "";
        let isViolation = false;
        let currentViolationType = "";

        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
            if (results.multiFaceLandmarks.length > 1) {
                isViolation = true;
                currentViolationType = "MULTIPLE FACES DETECTED!";
                aiAnalysis.multipleFacesFrames++;
                warningMsg = currentViolationType;
                if (!aiAnalysis.warnings.includes(currentViolationType)) {
                    aiAnalysis.warnings.push(currentViolationType);
                }
            } else {
                const landmarks = results.multiFaceLandmarks[0];

                const nose = landmarks[1];
                const leftEar = landmarks[234];
                const rightEar = landmarks[454];
                const yawRatio = Math.abs(nose.x - leftEar.x) / (Math.abs(nose.x - rightEar.x) + 0.001);

                const midEyeY = (landmarks[33].y + landmarks[263].y) / 2;
                const mouthY = (landmarks[13].y + landmarks[14].y) / 2;
                const noseY = landmarks[1].y;
                const pitchRatio = Math.abs(noseY - midEyeY) / (Math.abs(mouthY - noseY) + 0.001);

                const r_iris = landmarks[468];
                const r_top = landmarks[159];
                const r_bottom = landmarks[145];
                const r_inner = landmarks[133];
                const r_outer = landmarks[33];

                const l_iris = landmarks[473];
                const l_top = landmarks[386];
                const l_bottom = landmarks[374];
                const l_inner = landmarks[362];
                const l_outer = landmarks[263];

                const gazeH_Right = getHorizontalGazeRatio(r_iris, r_outer, r_inner);
                const gazeH_Left = getHorizontalGazeRatio(l_iris, l_inner, l_outer);

                const gazeV_Right = getVerticalGazeRatio(r_iris, r_top, r_bottom);
                const gazeV_Left = getVerticalGazeRatio(l_iris, l_top, l_bottom);

                const openRatioRight = getEyeOpenRatio(r_top, r_bottom, r_inner, r_outer);
                const openRatioLeft = getEyeOpenRatio(l_top, l_bottom, l_inner, l_outer);

                if (yawRatio < 0.8 || yawRatio > 3.2) {
                    isViolation = true;
                    currentViolationType = "PLEASE LOOK STRAIGHT!";
                }
                else if (pitchRatio < 1.0) {
                    isViolation = true;
                    currentViolationType = "HEAD TOO LOW!";
                } else if (pitchRatio > 1.6) {
                    isViolation = true;
                    currentViolationType = "HEAD TOO HIGH!";
                }
                else {
                    const isGlancingH = (gazeH_Right < 0.3 || gazeH_Right > 0.7) && 
                                        (gazeH_Left < 0.3 || gazeH_Left > 0.7);
                    
                    const isGlancingV = (gazeV_Right < 0.2 || gazeV_Right > 0.8) && 
                                        (gazeV_Left < 0.2 || gazeV_Left > 0.8);

                    if (isGlancingH) {
                        isViolation = true;
                        currentViolationType = "PLEASE LOOK AT CAMERA!";
                    } else if (isGlancingV) {
                        isViolation = true;
                        currentViolationType = "PLEASE LOOK AT CAMERA!";
                    }
                }
            }
        } else {
            isViolation = true;
            currentViolationType = "FACE NOT DETECTED!";
            aiAnalysis.noFaceFrames++;
            warningMsg = currentViolationType;
            if (!aiAnalysis.warnings.includes(currentViolationType)) {
                aiAnalysis.warnings.push(currentViolationType);
            }
        }
        if (isViolation) {
            if (violationStartTime === null) {
                violationStartTime = Date.now();
            }
            const violationDuration = Date.now() - violationStartTime;

            if (violationDuration >= VIOLATION_THRESHOLD_MS) {
                warningMsg = currentViolationType;
                if (!aiAnalysis.warnings.includes(currentViolationType)) {
                    aiAnalysis.warnings.push(currentViolationType);
                }
            }

            aiAnalysis.lookingAwayFrames++;

        } else {
            violationStartTime = null;
        }
        const warningEl = document.getElementById('ai-warning');
        if (warningMsg) {
            warningEl.textContent = warningMsg;
        } else {
            warningEl.textContent = "";
        }
    }

    faceMesh = new FaceMesh({locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
    }});
    
    faceMesh.setOptions({
        maxNumFaces: 2,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    
    faceMesh.onResults(onResults);

    camera = new Camera(videoElement, {
        onFrame: async () => {
            if (isAiRunning) {
                await faceMesh.send({image: videoElement});
            }
        },
        width: 640,
        height: 480
    });
    
    camera.start();
}