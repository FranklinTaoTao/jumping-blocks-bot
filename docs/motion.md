# Motion model

This document describes movement observed in recordings of 跃动方块 (Jumping
Blocks), followed by the approximation used by the bot. The game's source code
has not been inspected. Numerical values are measurements, not game constants
that are guaranteed across versions.

## Coordinates and units

Images are scaled to a board width of **420 units**. The board height depends on
its aspect ratio; the development recordings are approximately 744–745 units
high. Time is measured in seconds.

- `x` increases to the right.
- `Y` is vertical position in the world and increases downwards.
- `c` is the accumulated vertical camera offset.
- `y = Y + c` is vertical position on the screen.

Keeping world and screen coordinates separate matters: a character can climb
through the world while remaining at the same height on screen.

## Character motion

Once play begins, the character follows approximately ballistic arcs. A click
on either half of the board **replaces** both velocity components:

```text
left click:   vx = -speed, vy = -jump_speed
right click:  vx = +speed, vy = -jump_speed
```

The reset happens even in mid-air. A direction change therefore also creates a
new upward impulse. It is not a horizontal-only steering action.

Between clicks, the model is:

```text
x(t + dt)  = x(t) + vx * dt
Y(t + dt)  = Y(t) + vy * dt + 0.5 * gravity * dt²
vy(t + dt) = vy(t) + gravity * dt
```

The bundled calibration estimates:

| Quantity | Approximate value |
| --- | --- |
| Horizontal speed | 150 units/s |
| Upward reset speed | 705 units/s |
| Downward acceleration | 1,407 units/s² |
| Time from a reset to its apex | 0.50 s |
| Rise from one uninterrupted reset | 177 units |

Horizontal motion continues after the apex and while falling. Waiting lets the
current arc continue; it does not stop the character or make it hover. The
opening screen has an idle state before the first jump.

## Camera: a one-way lock at the middle

The observed behavior has three parts:

1. **Below the middle:** the character rises on screen, while obstacles and the
   background remain in the same screen positions.
2. **At the middle and still climbing:** the character stays near the middle;
   obstacles appear to move down as the camera follows upward progress.
3. **Falling:** the camera keeps its highest progress. The character falls down
   the screen; the scene does not scroll back to recover lost space.

A later jump from the lower part of the screen first raises the character
through the fixed scene. Scrolling resumes only after it reaches the middle
again and exceeds the previous high point.

With a screen anchor `m` near half the board height, this can be expressed as:

```text
c_next = max(c_previous, m - Y_next)
y_next = Y_next + c_next
```

Equivalently, camera offset depends on the lowest world `Y` reached so far. An
example with a rounded middle position of 370 illustrates a fall and a second
jump:

| Phase | World Y | Camera offset c | Screen y |
| --- | ---: | ---: | ---: |
| Below the middle | 550 | 0 | 550 |
| Rising | 400 | 0 | 400 |
| Reaches the middle | 370 | 0 | 370 |
| New upward progress | 300 | 70 | 370 |
| Falls | 400 | 70 | 470 |
| Jumps again, still below the middle | 350 | 70 | 420 |
| Returns to the previous high point | 300 | 70 | 370 |
| Makes further upward progress | 250 | 120 | 370 |

### The fall-off boundary also moves

If the board height is `H`, a character of half-height `hh` must satisfy:

```text
Y + c + hh < H
```

After climbing, the largest safe world Y becomes smaller. The planner carries
each candidate's highest point so that a later fall cannot use space already
lost offscreen. Safety margins make the implemented bound more conservative.

For example, a jump beginning at world Y=500 can peak near Y=325. With an anchor
of 370, the camera advances 45 units. A later position of world Y=719 is screen
Y=764, which is already outside a 740-unit board.

## Obstacles

### Bars and square blocks

The recorded bars and ordinary square blocks generally remain fixed in world
coordinates. Their apparent vertical motion comes from the camera:

```text
obstacle_screen_y = obstacle_world_Y + c
```

They stay fixed on screen during a fall or the below-middle portion of a jump.
When a new high point advances the camera, they move down on screen together.
This common displacement is also how the tracker estimates camera movement.

Observed examples include roughly 30-unit small squares and 96-unit large
squares. Bars and blocks can enter the top of the image partly clipped or be
partly covered by the score and promotional badge. The detector updates every
frame and retains useful partial contours rather than waiting for the whole
obstacle to become visible.

Small-block tracks can estimate translation independent of camera motion.
Unseen obstacle positions and future spawn patterns are not known in advance.

### Rotating hammers

Hammers have an irregular head-and-handle shape and rotate while camera motion
changes their screen position. Their collision region cannot be represented
accurately by a stationary axis-aligned rectangle alone.

The tracker estimates orientation from image moments, unwraps that orientation
across frames, and estimates angular speed. Once enough rotation is observed,
a rigid-motion fit estimates the pivot. A point is predicted approximately as:

```text
point(t + dt) = pivot + R(omega * dt) * (point(t) - pivot) + translation * dt
```

The initial model uses centroid motion until the pivot fit has enough evidence.
Collision checks use a convex hull, which conservatively fills the hammer's
inner notches. Magenta outlines in diagnostic recordings show its current and
predicted geometry. Partial visibility and changing contour shape can still
make these estimates uncertain.

## Clearing a wall

The planner checks the character's full rectangle throughout a predicted arc.
When its horizontal span overlaps a bar:

- Passing above requires the character's **bottom** to clear the bar's top.
- Passing below requires the character's **top** to remain below the bar's bottom.

This is why a direction change can be too early: it resets upward velocity and
may send the character into an underside, or toward a side before sufficient
height has been gained. Waiting also needs checking because horizontal motion
continues and can carry the character into an edge.

The implementation samples trajectories at intervals no longer than 15 ms and
inflates obstacles for clearance and command-timing uncertainty. This remains
a prediction based on the current scene, not a guarantee against new hazards.

## Render timing and observations

A mouse event and the first rendered frame showing its result are not
simultaneous. The recordings contain old falling positions shortly after a new
jump was pressed. Fitting those positions together with the new arc can make
upward velocity look much smaller than it really is.

The tracker therefore excludes the first 50 ms after a known click from a new
velocity fit and uses the measured reset model until three clean observations
are available. Camera displacement is subtracted before fitting world motion.
Future telemetry records press times, planning duration, and frame age so this
assumption can be measured again on another machine.

See [lessons learned](lessons-learned.md) for the recorded cases and
[`physics.py`](../jumpbot/physics.py), [`motion.py`](../jumpbot/motion.py), and
[`native_search.py`](../jumpbot/native_search.py) for the implementation.
