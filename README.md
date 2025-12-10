<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]


<br />
<div align="center">
  <a href="https://github.com/thinhngosy2060811/web-interview-recorder">
    <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Web Interview Recorder (Integrated)</h3>

  <p align="center">
    An AI-powered Asynchronous Video Interview Platform with real-time proctoring and automated scoring.
    <br />
    <a href="https://github.com/thinhngosy2060811/web-interview-recorder"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/thinhngosy2060811/web-interview-recorder">View Demo</a>
    &middot;
    <a href="https://github.com/thinhngosy2060811/web-interview-recorder/issues">Report Bug</a>
    &middot;
    <a href="https://github.com/thinhngosy2060811/web-interview-recorder/issues/new">Request Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#system-architecture">System Architecture</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#technical-specifications">Technical Specifications</a>
      <ul>
        <li><a href="#api-contract">API Contract</a></li>
        <li><a href="#file-handling">File Handling</a></li>
        <li><a href="#limits-reliability">Limits & Reliability</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

**Web Interview Recorder** is a comprehensive solution for automating the initial screening process in recruitment. It allows candidates to record video answers to interview questions while an AI engine monitors their behavior in real-time. On the backend, the system processes the video, transcribes the audio, and uses Generative AI to evaluate the candidate's performance.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### System Architecture & Flow

The system follows a Client-Server architecture designed to handle heavy media processing asynchronously:

1.  **Frontend (Client):** * Captures video stream and runs **MediaPipe FaceMesh** locally (Edge AI) to detect behavioral anomalies (looking away, multiple faces).
    * Manages recording logic and chunked uploads.
2.  **Backend (FastAPI):** * Acts as the orchestrator. It receives raw video streams, validates tokens, and dispatches background tasks.
3.  **Processing Layer:**
    * **FFmpeg:** Converts WebM blobs to MP4 for compatibility.
    * **Whisper AI:** Transcribes audio to text (ASR) in the background.
    * **Gemini AI:** Analyzes the transcription + behavioral metrics to grade the candidate.

> **⚠️ HTTPS Requirement:** Modern browsers (Chrome, Edge, Safari) strictly require a **Secure Context (HTTPS)** to access `navigator.mediaDevices.getUserMedia` (Camera & Microphone).
> * **Localhost:** Works by default (browsers treat `localhost` as secure).
> * **Production:** You **MUST** deploy this application with an SSL Certificate (HTTPS), otherwise the camera will not open.

### Built With

This project leverages a robust tech stack for high performance and AI integration:

* [![FastAPI][FastAPI]][FastAPI-url]
* [![Python][Python]][Python-url]
* [![JavaScript][JavaScript]][JavaScript-url]
* [![MediaPipe][MediaPipe]][MediaPipe-url]
* **OpenAI Whisper** (Speech-to-Text)
* **Google Gemini** (Generative AI Analysis)
* **FFmpeg** (Video Processing)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Technical Specifications

### 📡 API Contract

The backend exposes RESTful endpoints via FastAPI. Below are the core endpoints:

| Method | Endpoint | Description | Request Body | Response |
| :--- | :--- | :--- | :--- | :--- |
| **POST** | `/api/verify-token` | Verifies user role (Candidate/Admin) | `{ "token": "string" }` | `{ "ok": true, "role": "candidate" }` |
| **POST** | `/api/session/start` | Initializes interview session | `{ "token": "...", "userName": "..." }` | `{ "folder": "...", "questions": [...] }` |
| **POST** | `/api/upload-one` | Uploads video answer | `FormData`: video, token, analysisData | `{ "savedAs": "Q1.webm", "transcription": "processing" }` |
| **POST** | `/api/session/finish` | Finalizes session & grading | `{ "folder": "...", "questionsCount": 5 }` | `{ "ok": true }` |
| **GET** | `/api/admin/candidates` | Fetches list for dashboard | Query Param: `?token=...` | `{ "candidates": [...] }` |

### 📂 File Handling & Naming Convention

To ensure organized data storage, the system implements a strict naming strategy in the `uploads/` directory:

* **Session Folder:** `{DD}_{MM}_{YYYY}_{HH}_{MM}_{SanitizedUserName}`
    * *Example:* `25_10_2025_14_30_thinh_ngo`
* **Video Files:** * `Q{index}.webm` (Raw upload from browser)
    * `Q{index}.mp4` (Converted for compatibility)
* **Metadata:** `meta.json` stores user info, question list, and AI analysis results.

### 🛡️ Limits & Reliability

* **File Size Limit:** The system enforces a **50MB** hard limit per video upload (Configured in `config.py`). If a file exceeds this, a `413 Payload Too Large` error is returned.
* **Retry Policy:**
    * The frontend implements an **Exponential Backoff** retry mechanism in `main.js`.
    * If an upload fails (network error), it retries **3 times** (Wait times: 1s, 2s, 4s) before notifying the user.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

To get a local copy up and running, please follow these steps.

### Prerequisites

* **Python 3.9+**
* **FFmpeg**: Required for video processing.
    * *Windows:* `winget install ffmpeg` (verify with `ffmpeg -version`).
    * *Mac:* `brew install ffmpeg`.
    * *Linux:* `sudo apt install ffmpeg`.

### Installation

1.  **Clone the repo**
    ```sh
    git clone [https://github.com/thinhngosy2060811/web-interview-recorder.git](https://github.com/thinhngosy2060811/web-interview-recorder.git)
    cd web-interview-recorder
    ```

2.  **Create and Activate Virtual Environment**
    It is recommended to run this project in a virtual environment.
    * **PowerShell (Windows):**
        ```powershell
        python -m venv venv
        .\venv\Scripts\Activate.ps1
        ```
    * **Mac/Linux:**
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Python Packages**
    ```sh
    pip install fastapi uvicorn python-multipart python-dotenv requests openai-whisper google-generativeai pydantic
    ```

4.  **Configure Environment Variables**
    Create a `.env` file in the root directory and add your Google Gemini API Key:
    ```env
    GEMINI_API_KEY=your_actual_api_key_here
    ```

5.  **Project Structure Check**
    Ensure your directory structure looks like this so the module imports work correctly:
    ```text
    web-interview-recorder/
    ├── .env
    ├── app/               <-- Main package
    │   ├── __init__.py
    │   ├── main.py        <-- Entry point
    │   └── ...
    └── static/
    ```

6.  **Run the Server**
    Run the following command in **PowerShell** (at the root directory):
    ```powershell
    uvicorn app.main:app --reload
    ```
    The server will start at `http://127.0.0.1:8000`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

The system uses a **Token-based Authentication** mechanism (configured in `config.py`) to distinguish between Candidates and Admins.

### 1. For Candidates
* **Login:** Access the home page and enter a candidate token.
    * *Default Tokens:* `thinhbeo`, `thanhbusy`, `candidate`
* **Process:** Enter your name -> Grant Camera/Mic permissions -> Answer Questions.
* **Features:**
    * **Text-to-Speech:** Reads questions aloud.
    * **Countdown Timer:** Preparation and recording timers.
    * **Review:** Option to review and retry the recording once.

### 2. For Evaluators (Admins)
* **Login:** Access the home page and enter an admin token.
    * *Default Tokens:* `luandeptrai`, `hongraphay`, `admin456`
* **Dashboard:**
    * View a ranked list of candidates (High/Medium/Low priority).
    * Access detailed reports including Focus Score (%), Silence Ratio, and AI Summary.
    * Watch uploaded video responses.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Features

- [x] **Real-time AI Proctoring:** Detects head pose (yaw/pitch), eye gaze, and multiple faces using MediaPipe.
- [x] **Automated Workflow:** From recording -> upload -> processing -> grading.
- [x] **Speech-to-Text:** Integrated **OpenAI Whisper** for high-accuracy audio transcription.
- [x] **Smart Analytics:**
    - **Focus Score:** Tracks how often the candidate looks at the camera.
    - **Speech Rate (WPM):** Analyzes speaking speed (Slow/Good/Fast).
    - **Content Analysis:** Gemini evaluates the relevance and depth of the answer.
- [x] **Video Management:** Automatic WebM to MP4 conversion and file organization by session.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

**Project Contributors:**

* **Thinh Ngo** - [@thinhngosy2060811](https://github.com/thinhngosy2060811)
* **Hong Anh** - [@anhhongdangcode-pixel](https://github.com/anhhongdangcode-pixel)
* **Nuan** - [@nuan2779](https://github.com/nuan2779)
* **Fort** - [@Fort224](https://github.com/Fort224)

Project Link: [https://github.com/thinhngosy2060811/web-interview-recorder](https://github.com/thinhngosy2060811/web-interview-recorder)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

* [MediaPipe Face Mesh](https://developers.google.com/mediapipe/solutions/vision/face_mesh)
* [OpenAI Whisper](https://github.com/openai/whisper)
* [Google Gemini API](https://ai.google.dev/)
* [FastAPI Documentation](https://fastapi.tiangolo.com/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[contributors-shield]: https://img.shields.io/github/contributors/thinhngosy2060811/web-interview-recorder.svg?style=for-the-badge
[contributors-url]: https://github.com/thinhngosy2060811/web-interview-recorder/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/thinhngosy2060811/web-interview-recorder.svg?style=for-the-badge
[forks-url]: https://github.com/thinhngosy2060811/web-interview-recorder/network/members
[stars-shield]: https://img.shields.io/github/stars/thinhngosy2060811/web-interview-recorder.svg?style=for-the-badge
[stars-url]: https://github.com/thinhngosy2060811/web-interview-recorder/stargazers
[issues-shield]: https://img.shields.io/github/issues/thinhngosy2060811/web-interview-recorder.svg?style=for-the-badge
[issues-url]: https://github.com/thinhngosy2060811/web-interview-recorder/issues
[license-shield]: https://img.shields.io/github/license/thinhngosy2060811/web-interview-recorder.svg?style=for-the-badge
[license-url]: https://github.com/thinhngosy2060811/web-interview-recorder/blob/master/LICENSE.txt

[FastAPI]: https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi
[FastAPI-url]: https://fastapi.tiangolo.com/
[Python]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[JavaScript]: https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E
[JavaScript-url]: https://developer.mozilla.org/en-US/docs/Web/JavaScript
[MediaPipe]: https://img.shields.io/badge/MediaPipe-00A8E8?style=for-the-badge&logo=google&logoColor=white
[MediaPipe-url]: https://developers.google.com/mediapipe
