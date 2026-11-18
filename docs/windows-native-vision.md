# Native vision on Windows

`--native-vision` uses DXGI desktop duplication, a Direct3D 11 client-area
staging texture, and OpenCV detection in a C++ DLL. The Python module receives
timestamps, board dimensions, player and obstacle boxes, moving-shape hulls,
the score, and recognized menu actions. The live call returns no image buffer.
The Python tracker and Numba-compiled trajectory search consume the same
measurements as before.

## Build and run

Install the Visual Studio C++ desktop workload with the Windows SDK and CMake.
Extract the x64 Windows build of [OpenCV 5](https://github.com/opencv/opencv/releases/tag/5.0.0).
From the repository in PowerShell, point `OpenCV_DIR` to the extracted directory
containing `OpenCVConfig.cmake`.

```powershell
cmake -S native -B build/native-vision -A x64 "-DOpenCV_DIR=$env:OpenCV_DIR"
cmake --build build/native-vision --config Release
cmake --install build/native-vision --config Release --prefix (Join-Path $PWD 'build/native-vision/install')

.\.venv\Scripts\python.exe -m jumpbot live --native-vision --fps 90 --duration 20
```

Keep the game wholly inside one unrotated display. Add `--play` to enable
focused game input. Native vision uses the DXGI presentation timestamp and
defaults to `--render-delay 0`. For a custom install, set `JUMPBOT_NATIVE_DLL`
to the full path of `jumpbot_vision.dll` and keep its OpenCV DLL next to it.

With native vision, `--record` writes per-frame geometry, motion and decisions
to `frames.jsonl` and saves the score screen after each played round. `--show`
opens an OpenCV window in a separate process. It draws the player, planned path,
obstacles and their projected motion from geometry alone. The window updates at
up to 30 Hz and may skip display updates while recording retains every processed
frame. Move the preview beside the game and leave the game focused during play.
Closing the preview, pressing Escape or Q in it, or pressing F8 stops the run.
`--target` saves a screenshot directly from the DLL when reached. The standard
capture path still supports annotated diagnostic video.

## Verification

Compared with the DXGI and Python/OpenCV path, native detection matched all
1,219 sampled replay frames for board ROI, player, obstacle boxes, shape hulls,
score and menu decisions. In the 200-point recording, median gameplay detection
including OCR took 2.45 ms with Python/OpenCV and 1.46 ms with native vision.

In a separate no-input test on the open game window, both backends processed
about 275 distinct presentations per 4.7-second sample. The median age of a
frame when detections were available was 7.0–7.5 ms with the DXCam path and
5.2 ms with the native path. Capture polling and detection took 3.7 ms at the
median on a successful native read. The native benchmark used a passive screen
and did not measure scores or gameplay latency after an input.

Run the parity and timing test against a local annotated recording by
cropping to its original-frame pane. Set the crop to match your input video.

```powershell
.venv/Scripts/python.exe tools/bench_native_vision.py $env:JUMPBOT_BENCH_VIDEO --crop 0 0 420 785 --width 722 --stride 9
```
