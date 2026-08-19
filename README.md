# Exam monitor — backend

FastAPI backend running the three detection modules (M1 object detection,
M2 pose estimation, M3 head/gaze tracking), a rule-based fusion engine that
decides when a signal becomes a logged alert, and the WebSocket endpoint
the Next.js frontend connects to.

## 1. Install Python

You need Python 3.10+. Check what you have:

```bash
python3 --version
```

If you don't have it, download from https://www.python.org/downloads/ (Windows/macOS)
or use your package manager on Linux (`sudo apt install python3 python3-venv` on Ubuntu/Debian).

## 2. Create a virtual environment

This keeps this project's packages separate from anything else on your machine.
From inside this `backend/` folder:

```bash
python3 -m venv venv
```

Activate it:

```bash
# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

You'll know it worked because your terminal prompt will show `(venv)` at the start.
You need to activate this every time you open a new terminal to work on the project.

## 3. Install the dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, OpenCV, Ultralytics (YOLOv8), MediaPipe, and the
MongoDB/Cloudinary clients. It'll take a few minutes the first time —
Ultralytics and MediaPipe are the largest.

**First-run downloads**: the first time you actually run the app, two things
download automatically in the background — YOLOv8n's weights (~6MB) and
MediaPipe's pose/face landmark models (a few MB each). This needs normal
internet access and only happens once; after that they're cached locally.

## 4. Set up your credentials

Copy the example env file:

```bash
cp .env.example .env
```

Then open `.env` and fill in two things:

**MongoDB Atlas** (free tier is enough):
1. Go to https://www.mongodb.com/cloud/atlas/register and create a free account
2. Create a free (M0) cluster
3. Under "Database Access," create a user with a username/password
4. Under "Network Access," add `0.0.0.0/0` (allow from anywhere) — fine for a
   student project; you can tighten this later
5. Click "Connect" > "Drivers" and copy the connection string into `MONGODB_URI`

**Cloudinary** (free tier is enough):
1. Go to https://cloudinary.com/users/register/free and create a free account
2. Your dashboard homepage shows Cloud Name, API Key, and API Secret directly —
   copy those three into `.env`

## 5. Run it locally

```bash
uvicorn app.main:app --reload --port 8000
```

You should see it start up and log that it's listening on `http://localhost:8000`.
Visit `http://localhost:8000/health` in a browser — you should see `{"status":"ok"}`.

Now point the frontend at it: in the frontend's `.env.local`, set
`NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/monitor` and run the frontend's
`npm run dev`. Open the frontend page and allow camera access — frames should
start flowing to this backend and you'll see detection activity in this
terminal's logs.

## 6. Deploying to Render

1. Push this `backend/` folder to a GitHub repo (its own repo, or a subfolder
   with Render's "Root Directory" setting pointed at it)
2. On Render: New > Web Service > connect the repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add the same environment variables from your `.env` file under Render's
   "Environment" tab (Render does not read your local `.env` file — you
   re-enter the values in its dashboard)
6. Once deployed, your WebSocket URL will be
   `wss://your-service-name.onrender.com/ws/monitor` — put that in the
   frontend's `NEXT_PUBLIC_WS_URL` for production

**Note**: Render's free tier spins down after inactivity and takes ~30-60
seconds to wake back up on the next request. Fine for development/demo;
worth knowing before your defense so you hit the URL a minute early to wake it up.

## What's here

```
app/
  main.py                  — FastAPI app + the /ws/monitor WebSocket endpoint
  config.py                — all settings, loaded from environment variables
  fusion.py                — combines M1/M2/M3 output, tracks persistence, decides alerts
  detectors/
    objects.py              — M1: YOLOv8n object detection (phone, book, person)
    pose.py                  — M2: MediaPipe Pose (hand-drop / lean detection)
    gaze.py                   — M3: MediaPipe Face Mesh + solvePnP (head turn detection)
  storage/
    cloudinary_client.py      — uploads alert snapshots, returns a hosted URL
    mongo_client.py            — inserts/reads alert documents
```

## Tuning detection sensitivity

All the thresholds live in `.env` (loaded via `config.py`), not hardcoded in
the detector files — adjust these once you're testing with a real camera angle:

- `OBJECT_CONFIDENCE_THRESHOLD` — how confident YOLO must be before counting a
  phone/book detection (raise if you get false positives, lower if it's missing things)
- `ALERT_PERSISTENCE_FRAMES` — how many consecutive ~1-second checks a signal
  must persist before it's logged as an alert (raises this to reduce false alarms
  from brief, incidental movement)
- `YAW_THRESHOLD_DEGREES` in `gaze.py` — how far a head must turn before counting
  as "looking away" (this one's in the file directly since it's a detection-specific
  constant rather than a deployment setting)

## Known simplification worth knowing about

`pose.py` and `gaze.py` currently track one person at a time (MediaPipe's
default single-person mode). For a genuinely multi-student wide-angle shot,
the more complete approach is to first get each person's bounding box from
`objects.py` (YOLO's "person" class), crop the frame to each box, and run
pose/gaze on each crop separately. This is flagged here rather than silently
built in because it adds real complexity (per-person tracking across frames)
— worth doing once the single-person pipeline is proven working end-to-end,
not before.

## Docker / Render

Quick steps to build and run locally with Docker:

```bash
docker build -t exam-monitor-backend:latest .
docker run --rm -p 8000:8000 \
  -e MONGODB_URI='your-mongo-uri' \
  -e CLOUDINARY_CLOUD_NAME='...' \
  -e CLOUDINARY_API_KEY='...' \
  -e CLOUDINARY_API_SECRET='...' \
  -e PORT=8000 \
  exam-monitor-backend:latest
```

Deploying to Render:

- Create a GitHub repo containing this project and connect it to Render.
- Render will detect the `Dockerfile` and build using it; set environment
  variables in the Render dashboard (same names as in `.env`).
- Ensure the start command is `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

If you prefer Render's managed Python build instead of Docker, set the
build command to `pip install -r requirements.txt` and the start command
to the `uvicorn` line above; however the Dockerfile approach ensures
system dependencies for OpenCV are present and is recommended.

