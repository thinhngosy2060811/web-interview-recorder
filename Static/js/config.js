let faceMesh;
let camera;
let aiAnalysis = {
    totalFrames: 0,
    lookingAwayFrames: 0,
    multipleFacesFrames: 0,
    noFaceFrames: 0,
    warnings: []
};

const TIME_LIMIT_SEC = 180; 
let QUESTIONS = []; 
let timerInterval;

let retriesUsed = 0;
let isAiRunning = false;
let pendingVideoBlob = null;
let retryCountForCurrentQuestion = 0;
const MAX_RETRIES_PER_QUESTION = 1;
let violationStartTime = null;
let countdownInterval;
let currentUtterance = null;
let prepInterval;

const VIOLATION_THRESHOLD_MS = 5000;
const API_BASE = '';
const MAX_RETRIES = 3;
const MAX_FILE_SIZE = 50 * 1024 * 1024;

let token = '';
let userName = '';
let folder = '';
let currentQuestionIndex = 0;
let mediaRecorder = null;
let recordedChunks = [];
let stream = null;
let uploadRetryCount = 0;