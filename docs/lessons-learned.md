# Lessons learned from development and recordings

These notes distinguish observed failures, implemented changes, and validation
limits. A development build reached 100 in a live game. That is a verified
milestone, not a measured success rate for every run or every later revision.

## 1. Input latency is part of the model

The initial interaction plan was to identify the character and obstacles, fit a
parabola, and choose left or right. A useful trajectory can become stale while
capture, planning, and input delivery take place.

**Change:** capture directly into memory, avoid queued input frames, include
measured planning time in latency estimates, timestamp the mouse press rather
than its release, and reject stale decisions. Video encoding uses a separate,
bounded output queue.

**Lesson:** fast vision alone is insufficient; the predicted position must
correspond to when a click can actually take effect.

## 2. A click resets both direction and upward velocity

Clicks do not add a small steering correction to an existing arc. Each creates
a new upward-left or upward-right velocity. Frequent resets can cause rapid
ascent and leave too little time to align with the next opening.

**Change:** search over waiting and future resets, check the entire character
rectangle, and include a penalty for unnecessary clicks. Timing margins also
avoid routes that require a nearly perfect last-moment reversal.

**Lesson:** a turn needs enough vertical clearance as well as horizontal room.
This was visible in a 22-point run that eventually jumped into a bar's underside.

## 3. Screen motion and world motion are different

During climbing, the character can remain near the middle while the scene
scrolls. During a fall, that scroll is retained. A jump from the lower screen
first moves through a fixed scene before scrolling resumes at the middle.

**Change:** subtract camera motion when estimating velocity and carry each
planned path's highest point to update its future fall-off boundary.

**Validation:** regression tests cover the below-middle phase, scrolling,
falling, a second jump, and a fall into space lost by a previous climb. The
camera-boundary omission was confirmed in the model; it was not established
as the cause of the two deaths in the latest reviewed batch.

See [the motion description](motion.md) for equations and an example sequence.

## 4. UI overlays can hide physical obstacles

A bar connected to a black score numeral failed a simple rectangular-contour
filter. A promotional badge also made the visible beginning of a bar look much
shorter. Re-estimating the board boundary while a bar crossed its top could
shift the coordinate system itself.

**Change:** detect edge bars row by row, conservatively bridge the known badge
region, retain partially covered large squares, and keep the board boundary
fixed until the window size changes.

**Lesson:** visual irregularity does not imply a physical opening, especially
under a HUD element. Observations need to be interpreted in context.

## 5. A partly visible hammer is already useful evidence

In the reviewed 83-point run, an incoming hammer was discarded while its
contour touched the top of the image. The original detector first reported it
at about 79.534 seconds. Retaining the clipped contour reported it at about
79.393 seconds: **approximately 141 ms earlier** on the same saved frames.

**Change:** accept useful central contours clipped by the top boundary.

**Validation limit:** this proves earlier detection on that recording. It does
not prove that the changed controller would have survived the encounter.

## 6. Rendered positions can straddle a jump reset

Old falling frames can be included in the fit for a newly pressed jump. This
can make the estimated upward speed much lower than the reset model predicts.

**Change:** leave the first 50 ms after a known click out of the new arc fit and
use the reset prior until enough clean observations are available.

**Offline result:** on a held-out position about 102 ms later, the vertical
forecast error decreased from approximately **25 to 3 normalized pixels**.

**Validation limit:** this is one recorded regression case. The delay can vary
with rendering, hardware, and load; live score improvement remains unmeasured.

## 7. Shape and outfit changes should not erase tracking

The original player detector depended on a red headband. Unlocking a yellow
outfit made it lose the character. Rotating T-shaped hammers also failed the
solid-rectangle test used for ordinary blocks.

**Change:** identify colored headbands together with body geometry; use hulls,
orientation tracking, and pivot estimates for irregular moving hazards. Known
outfit dialogs have their own UI handler.

**Lesson:** separate the stable role of an object from one appearance or pose.

## 8. A longer nominal horizon is not a safety guarantee

A diagnostic label could report three predicted seconds shortly before a
collision. That prediction assumed the current detections, velocity estimate,
and timing model. A missed hammer or a bad initial velocity invalidated it.

**Change:** label the output as a prediction, record a numeric horizon, and log
actual press times, frame age, score reads, and planning latency. The diagnostic
HUD now sits below the board so it cannot cover incoming hazards.

**Lesson:** investigate prediction error and observation quality before simply
increasing the search horizon.

## 9. Replay evidence and live evidence answer different questions

A saved recording contains the actions that actually happened. A different
suggested action does not change the rest of that recording into a simulation
of the alternative outcome.

**Practice:** use recordings to inspect detections and held-out predictions;
use live runs to establish scores. Separate completed attempts from a session
that ends while still active, and exclude an old score screen at startup from
new-round high-score calculations.

## Windows lessons

### Calibrate reset latency in each session

Capture, planning, button delivery, and the visible velocity reset have
different timestamps. A fixed input-delay setting from an earlier session can
put a valid route on the wrong side of a bar. Estimate the current delay from
observed horizontal reversals and track its spread. Use the adaptive setting
with aligned delivery when the response is uncertain, and compare each
estimate with later observations before revising the gravity or launch speed.

### Plan the exit and leave room for timing error

Use the next visible gap to choose where to leave the current opening. Wall
contact stops horizontal movement, so repeatedly jumping on the same side can
consume several launch windows without advancing. A safer route can involve
coasting lower before turning inward. Recheck any queued reversal on fresh
geometry and keep the reset inside a verified delivery window. If a slightly
later jump would hit a bar or block, prefer a wider route when one is available.

### Track obstacles in camera-corrected coordinates

Separate camera easing from player and obstacle motion before fitting
velocity. Keep the detected outline as the forecast starts, especially at
slider and hammer reversals. A clipped contour or score overlap can bias the
fit. Leave extra clearance until fresh frames establish the motion.

### Verify with the right evidence

Use saved observations to diagnose a collision and simulated courses to
compare planner changes under timing jitter. Verify scores during live play.
Record frame age, control timing, focus pauses, and missed captures so a
missing observation can be distinguished from a poor route. Keep raw gameplay
and debug captures local.
