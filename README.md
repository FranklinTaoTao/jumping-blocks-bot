# Jumping Blocks Bot · 跃动方块


*A recorded development run: game on the left; obstacle detection, hammer
geometry, and trajectory predictions on the right.*

A real-time bot for **Jumping Blocks (跃动方块)**, built with OpenCV, ballistic
prediction, and deterministic planning. No machine learning or game-memory
access is used.

The bot detects the character, bars, blocks, and rotating hammers, then plans
left/right clicks through the visible obstacles. A development build reached
**100 points** in a live game. More recent camera and timing fixes have passed
offline regression tests; they have not yet been validated in another live run.

[Motion description](docs/motion.md) · [Lessons learned](docs/lessons-learned.md)

## Requirements

- Windows, or Linux with an **X11** desktop session; Wayland live capture is not supported.
- Python **3.12 or newer**; Linux also requires the system's `libX11` library.
- The game open in a visible, unobstructed WeChat mini-app window.

## Install

```bash
git clone git@github.com:FranklinTaoTao/jumping-blocks-bot.git
cd jumping-blocks-bot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Runtime dependencies are declared in `pyproject.toml`. Installing with
`python -m pip install -r requirements.txt` is also supported.

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\run.ps1 --show
.\run.ps1 --play --focus --duration 60
```

Windows selects the visible portrait game by its title (`跃动方块`). Use
`--title` if the title differs. Capture reads the client area from the desktop
using MSS, so keep the game unobstructed and on screen. Input uses Windows
SendInput and pauses when the game loses focus. Run the game without
administrator rights so Windows permits input from the bot.

The Windows `--native-vision` option keeps capture and detection inside a C++
DLL and passes detections to the Python planner. See the [native build and
benchmarks](docs/windows-native-vision.md).

The MSS path defaults to `--render-delay 0.040`. Subsequent controlled
tests found that a fixed display delay did not transfer consistently between
runs. The experimental configuration below uses spatial parabola fitting and
no additional fixed display delay.
Live clocks use the high-resolution performance counter, and compiled search
warms up with the same argument types as live play.

See the [Windows lessons learned](docs/lessons-learned.md#windows-lessons) for calibration and timing guidance.

```powershell
.\run.ps1 --play --focus --physics jumpbot/assets/physics_windows_spatial.json `
  --render-delay 0 --adaptive-motion --camera-tau 0.011 `
  --search-step 0.025 --beam 600 --cooldown 0.18 `
  --target 200 --duration 600 --record artifacts/windows-trial
```

## Preview and play

```bash
# Preview detections and predictions without clicking.
jumping-blocks-bot live --show

# Play for up to one minute; focus the game to arm input.
jumping-blocks-bot live --play

# Record up to three rounds, stopping early if a live score reaches 200.
jumping-blocks-bot live --play --focus --rounds 3 --duration 600 \
  --target 200 --record artifacts/trial
```

`python -m jumpbot` provides the same commands. From a source checkout,
`./run.sh --play` is a shortcut that uses `.venv`.

**Stop with F8**, Ctrl+C in the launching terminal, or by moving the pointer to
the screen's top-left corner. Switching away from the game pauses clicks.
`--focus` focuses the game once at startup. The preview window may take focus;
select the game again to allow play. Q or Escape also closes the preview.

Useful options:

| Option | Behavior |
| --- | --- |
| `--title 跃动方块` | Select a game when multiple WeChat mini-app windows are open. |
| `--roi x,y,width,height` | Override automatic board detection in source pixels. |
| `--duration 120` | Limit the session to 120 seconds. |
| `--no-auto-restart` | Stop at the score screen. |
| `--physics path/to/calibration.json` | Use a custom calibration. |
| `--planner python` | Use the narrower Python reference search. |
| `--record artifacts/trial` | Save a diagnostic video and frame telemetry. |

Live mode defaults to preview: mouse clicks require `--play`. During play, known
score screens restart automatically. The bot declines helmet and revive offers,
closes recognized ads using their X, and accepts a newly unlocked outfit using
the game's “Use now” button. Unknown dialogs receive no automatic UI clicks.

## How it works

1. **Capture:** native `XGetImage` returns window pixels directly into NumPy and
   OpenCV. Input frames never pass through screenshot files or a frame queue.
2. **Detect:** color and contour geometry locate the character and obstacles.
   Edge-row detection keeps bars visible when score text touches them, and
   partially visible incoming hazards are retained.
3. **Track:** static geometry estimates camera scrolling. Several observations
   estimate velocity; timestamped clicks replace velocity with an upward-left
   or upward-right reset. Rotating shapes use hull and pivot estimates.
4. **Plan:** a visibility graph guides a compiled beam search over up to 900
   routes and three seconds. Intermediate collision checks include the full
   character rectangle, rotating hulls, and command-timing uncertainty.
5. **Act:** XTEST sends short press/release clicks inside the appropriate half
   of the board. Focus, geometry, frame age, and stop controls gate input.

The camera is modeled separately from physics. A character below the middle
rises while the scene stays fixed. Further climbing scrolls the scene, and
falling does not undo that progress. Every candidate therefore carries its
highest point so it cannot plan a retreat into space already lost offscreen.

All measurements use a normalized board width of 420 units. The bundled
calibration is approximately 150 units/s sideways, a 705 units/s upward reset,
and 1,407 units/s² of gravity. Calibration files and UI templates load once at
startup. Recording encodes output on a bounded background queue; an overloaded
encoder drops output frames rather than delaying input capture.

## Offline calibration and review

These commands read saved files and do not control the live game:

```bash
jumping-blocks-bot calibrate /path/to/recording.webm \
  --output artifacts/physics.json

jumping-blocks-bot replay /path/to/recording.webm \
  --physics artifacts/physics.json --output artifacts/replay.mp4

python tools/review_recording.py artifacts/trial --output artifacts/review
```

Calibration removes camera motion before fitting clean quadratic arcs and uses
container timestamps for variable-frame-rate recordings. Replay checks
perception and suggested actions; it cannot establish a counterfactual score.

Recordings contain the original window beside detection and trajectory
overlays. HUD text sits below the board so incoming hazards stay visible.
Telemetry includes actual press times, planning latency, score reads, and the
nominal predicted horizon. The horizon depends on current detections and motion
estimates; it does not guarantee safety against unseen obstacles.

## Development

```bash
python -m pip install -e '.[dev]'
python -m ruff check .
python -m ruff format --check .
python -m unittest discover -s tests -v
python -m build
```

Tests run offline and cover detection, motion, both planners, camera behavior,
UI recognition, and regressions extracted from recorded failures. GitHub
Actions runs the same checks and builds the package without accessing a game.

| Module | Purpose |
| --- | --- |
| `jumpbot/cli.py` | Live, calibration, and replay commands. |
| `jumpbot/capture.py`, `jumpbot/x11.py` | In-memory X11 capture and gated mouse input. |
| `jumpbot/vision.py`, `jumpbot/motion.py` | Detection and rotating-hazard tracking. |
| `jumpbot/physics.py` | Ballistics, velocity fitting, and camera projection. |
| `jumpbot/navigation.py` | Geometric route guidance. |
| `jumpbot/fast_planner.py`, `jumpbot/native_search.py` | Compiled trajectory search. |
| `jumpbot/planner.py` | Python reference planner. |
| `jumpbot/render.py`, `jumpbot/diagnostics.py` | Overlays and asynchronous output recording. |
| `jumpbot/assets/` | UI templates, numeral templates, and default calibration. |

## Current limits

The controller still collides sometimes. Newly appearing hazards, imperfect
motion estimates, rendering delays, and changed UI layouts can invalidate a
prediction. A verified development run measured roughly 26 ms median / 39 ms
p95 for the capture/decision/input loop on the development machine; performance
depends on hardware, window size, and scene complexity.

Recordings, screenshots, generated reviews, virtual environments, and compiled
caches are excluded from version control. The Windows branch uses synthetic inputs for its offline checks.
