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

In the 22-point run, old falling frames were included in the fit for a newly
pressed jump. Around 109.209 seconds, the logged upward speed was approximately
357 units/s even though the reset model indicated a substantially faster rise.

**Change:** leave the first 50 ms after a known click out of the new arc fit and
use the reset prior until enough clean observations are available.

**Offline result:** on a held-out position about 102 ms later, the vertical
forecast error decreased from approximately **25 to 3 normalized pixels**.
The numerical observation sequence is preserved in
[`tests/fixtures/late_jump.json`](../tests/fixtures/late_jump.json).

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

The reviewed batch contained completed scores of 83 and 22, followed by an
unfinished attempt paused at 49. The later fixes were validated offline while
live play remained stopped.

## Next validation

The next authorized live session should compare the revised controller with the
recorded baseline, check whether early hammer detections remain stable, and
measure jump-reset delay across more events. No further score claim follows
from the offline fixes alone.
