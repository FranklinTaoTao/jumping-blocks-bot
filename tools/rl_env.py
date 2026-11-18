"""Offline training plant; measured Windows dynamics, never controls the game.

Procedural training courses and reconstructed recording courses are separate.
This approximate plant is not a replacement for pixel-pipeline or live validation.
"""

import numpy as np
from numba import njit, prange

DT = 0.005
GLOBALS = 8
OBJECTS = 12
FEATURES = 7
OBS_SIZE = GLOBALS + OBJECTS * FEATURES


@njit(cache=True, parallel=True)
def _tracked_globals(s, params, history, heads, samples, sample_heads, counts, camera, obs):
    """Training surrogate for the runtime tracker's quantization and arc fit."""
    for e in prange(len(s)):
        if s[e, 13] < 0:
            continue
        t = s[e, 10]
        height = params[e, 3]
        delay = int(params[e, 6] / DT + 0.5)
        visual = history[e, (heads[e] - delay) % 32]
        x, y, _, _, peak, scene = visual[:6]
        target = max(0.0, height / 2 - peak)
        px = min(420.0, max(0.0, round(x - 15.5) + 15.0))
        py = round(y + target - 15) + 15.0
        scene = round(scene)
        relative = py - scene
        count = counts[e]
        dt = t - camera[e, 0]
        if count and dt > 1e-6:
            shift = scene - camera[e, 1]
            camera[e, 2] = 0.6 * camera[e, 2] + 0.4 * max(0.0, shift / dt)
        camera[e, 0] = t
        camera[e, 1] = scene
        previous = samples[e, sample_heads[e]]
        raw_vy = (relative - previous[2]) / dt + 1415 * dt / 2 if count and dt > 1e-6 else 0.0
        sample_heads[e] = (sample_heads[e] + 1) % 8
        samples[e, sample_heads[e], 0] = t
        samples[e, sample_heads[e], 1] = px
        samples[e, sample_heads[e], 2] = relative
        counts[e] = count + 1
        last_click = t - s[e, 6] + 0.035 if s[e, 8] else -100.0
        n = 0
        sum_t = sum_y = sum_tt = sum_ty = 0.0
        for k in range(min(8, count + 1)):
            point = samples[e, (sample_heads[e] - k) % 8]
            if point[0] >= last_click + 0.050:
                ts = point[0] - t
                adjusted = point[2] - 1415 * ts * ts / 2
                n += 1
                sum_t += ts
                sum_y += adjusted
                sum_tt += ts * ts
                sum_ty += ts * adjusted
        denominator = n * sum_tt - sum_t * sum_t
        vy = raw_vy
        if n >= 3 and denominator > 1e-10:
            vy = (n * sum_ty - sum_t * sum_y) / denominator
        elif s[e, 8]:
            vy = -688.55 + 1415 * max(0.0, t - last_click)
        vx = s[e, 8] * 150.4
        if (px <= 1 and vx < 0) or (px >= 419 and vx > 0):
            vx = 0.0
        obs[e, 0] = px / 210 - 1
        obs[e, 1] = py / height * 2 - 1
        obs[e, 2] = vx / 150
        obs[e, 3] = vy / 700
        obs[e, 7] = camera[e, 2] / 700
        if count + 1 < 3 or t < last_click:
            obs[e, 4] = 0.0


@njit(cache=True, parallel=True)
def _observe(s, params, boxes, history, heads, obs, observation_version):
    for e in prange(len(s)):
        if s[e, 13] < 0:
            continue
        height = params[e, 3]
        delay = int(params[e, 6] / DT + 0.5)
        h = history[e, (heads[e] - delay) % 32]
        x, y, vx, vy, peak, scene, target_speed, scene_speed = h
        target = max(0.0, height / 2 - peak)
        screen_y = y + target
        relative_y = screen_y - scene
        if observation_version >= 3:
            x = min(420.0, max(0.0, round(x - 15.5) + 15.0))
            screen_y = round(screen_y - 15) + 15.0
        obs[e, :] = 0.0
        obs[e, 0] = x / 210 - 1
        obs[e, 1] = screen_y / height * 2 - 1
        obs[e, 2] = vx / 150
        obs[e, 3] = (vy + target_speed - scene_speed) / 700
        obs[e, 4] = min(1.0, max(0.0, s[e, 6]) / 0.12)
        if observation_version >= 2:
            obs[e, 5] = s[e, 8]  # Last direction also identifies the idle start.
        if s[e, 9] > 0:
            obs[e, 5] = s[e, 8]
            obs[e, 6] = min(1.0, s[e, 9] / 0.1)
        obs[e, 7] = scene_speed / 700
        # Keep nearest visible objects, independent of their storage order.
        distances = np.full(OBJECTS, 1e20)
        for k in range(boxes.shape[1]):
            bx, by, bw, bh = boxes[e, k]
            if bw <= 0 or by + scene + bh < 0 or by + scene > height:
                continue
            if observation_version >= 3:
                left, right = max(0.0, round(bx)), min(420.0, round(bx + bw))
                top = max(0.0, round(by + scene))
                bottom = min(height, round(by + scene + bh))
                edge = bx < 4 or bx + bw > 416
                bw, bh = right - left, bottom - top
                if bw <= 0 or bh < (6 if edge else 9):
                    continue
                if not edge:
                    fill = (bw - 1) * (bh - 1) / (bw * bh)
                    occluded_square = top < 130 and bw >= 50 and 0.8 < bw / bh < 1.25
                    if fill < 0.78 or (top < 130 and fill < 0.90 and not occluded_square):
                        continue
                    if left >= 7 and right <= 135 and top >= 54 and bottom <= 101:
                        continue
                bx = left
                by = top - scene
                relative_y = screen_y - scene
            distance = abs(by + bh / 2 - relative_y)
            slot = -1
            worst = -1.0
            for j in range(OBJECTS):
                if distances[j] > worst:
                    worst = distances[j]
                    slot = j
            if distance < worst:
                distances[slot] = distance
                o = GLOBALS + slot * FEATURES
                obs[e, o] = (bx + bw / 2 - x) / 210
                obs[e, o + 1] = (by + bh / 2 - relative_y) / 420
                obs[e, o + 2] = bw / 210
                obs[e, o + 3] = bh / 420
                obs[e, o + 4] = 0
                obs[e, o + 5] = 0
                obs[e, o + 6] = 1


@njit(cache=True, parallel=True)
def _step(s, params, boxes, history, heads, actions, rewards, done, reasons):
    for e in prange(len(s)):
        if s[e, 13] < 0:
            rewards[e] = 0.0
            continue
        speed, jump, gravity, height, tau, period, presentation, input_lag, process = params[e]
        old_peak = s[e, 4]
        reward = 0.0
        reason = 0
        if actions[e] and s[e, 6] >= 0.12 and s[e, 7] <= 0:
            s[e, 7] = process + input_lag
            s[e, 8] = -1 if actions[e] == 1 else 1
            s[e, 9] = process + 0.035
            s[e, 6] = -process
            reward -= 0.001
        # Broad phase once per control interval; even the fastest trajectory
        # cannot reach a box outside this vertical neighborhood in 60 ms.
        near = np.empty(32, np.int64)
        count = 0
        relative_y = s[e, 1] + max(0.0, height / 2 - s[e, 4]) - s[e, 5]
        for k in range(boxes.shape[1]):
            b = boxes[e, k]
            if b[2] > 0 and b[1] < relative_y + 200 and b[1] + b[3] > relative_y - 200:
                if count < 32:
                    near[count] = k
                    count += 1
        for micro in range(int(period / DT + 0.5)):
            if s[e, 7] > 0:
                if s[e, 7] <= 1e-9:
                    s[e, 7] = 0
                    s[e, 2] = s[e, 8] * speed
                    s[e, 3] = -jump
                    s[e, 13] = 1
                else:
                    s[e, 7] = max(1e-12, s[e, 7] - DT)
            s[e, 9] = max(0.0, s[e, 9] - DT)
            before = max(0.0, height / 2 - s[e, 4])
            s[e, 0] = min(420.0, max(0.0, s[e, 0] + s[e, 2] * DT))
            if s[e, 0] <= 0 or s[e, 0] >= 420:
                s[e, 2] = 0  # Nonfatal sidewalls; vertical motion continues.
                reward -= 0.25 * DT
            if s[e, 13] > 0:
                s[e, 1] += s[e, 3] * DT + gravity * DT * DT / 2
                s[e, 3] += gravity * DT
            s[e, 4] = min(s[e, 4], s[e, 1])
            target = max(0.0, height / 2 - s[e, 4])
            scene_before = s[e, 5]
            rate = (target - before) / DT
            if tau > 1e-6:
                decay = np.exp(-DT / tau)
                s[e, 5] = target - rate * tau + (scene_before - before + rate * tau) * decay
            else:
                s[e, 5] = target
            heads[e] = (heads[e] + 1) % 32
            h = history[e, heads[e]]
            h[0:6] = s[e, 0:6]
            h[6] = rate
            h[7] = (s[e, 5] - scene_before) / DT
            s[e, 6] += DT
            s[e, 10] += DT
            s[e, 11] += DT
            screen_y = s[e, 1] + target
            relative_y = screen_y - s[e, 5]
            if screen_y + 15 >= height:
                reason = 1
            for j in range(count):
                bx, by, bw, bh = boxes[e, near[j]]
                if (
                    s[e, 0] + 15.5 > bx
                    and s[e, 0] - 15.5 < bx + bw
                    and relative_y + 15 > by
                    and relative_y - 15 < by + bh
                ):
                    reason = 1
            if reason:
                reward -= 2.5
                break
            if s[e, 4] < s[e, 12] - 10:
                s[e, 12] = s[e, 4]
                s[e, 11] = 0
            if s[e, 11] > 12:
                reason = 2
                reward -= 2.0
                break
            if relative_y + 15 < s[e, 14]:
                reason = 3
                reward += 3.0
                break
            if s[e, 10] > 160:
                reason = 4
                break
        reward += (old_peak - s[e, 4]) / 420 - 0.01 * period
        s[e, 15] += reward
        rewards[e] = reward
        reasons[e] = reason
        done[e] = reason != 0


class CourseEnv:
    """Vector plant with deterministic seeds and independently generated courses."""

    def __init__(
        self,
        n=64,
        seed=142,
        stage=0,
        courses=None,
        tiles=48,
        difficulty=1.0,
        observation_version=2,
        big_start=23,
        pixel_envs=0,
        processing=0.010,
        large_x=(140, 185),
        pair_probability=0.65,
    ):
        self.n, self.stage, self.tiles = n, stage, tiles
        self.difficulty = difficulty
        self.observation_version = observation_version
        self.big_start = big_start
        self.processing = processing
        self.large_x = large_x
        self.pair_probability = pair_probability
        self.pixel_envs = min(n, pixel_envs)
        self.pixel_trackers = [None] * self.pixel_envs
        self.pixel_detectors = [None] * self.pixel_envs
        if self.pixel_envs:
            import cv2

            from tools.offline_course_bench import assets

            cv2.setNumThreads(1)
            self.pixel_assets = assets()
        self.rng = np.random.default_rng(seed)
        self.courses = courses
        self.boxes = np.zeros((n, max(tiles * 6, 512), 4), np.float64)
        self.s = np.zeros((n, 16), np.float64)
        self.params = np.zeros((n, 9), np.float64)
        self.history = np.zeros((n, 32, 8), np.float64)
        self.heads = np.zeros(n, np.int64)
        self.obs = np.zeros((n, OBS_SIZE), np.float32)
        self.sensor_samples = np.zeros((n, 8, 3), np.float64)
        self.sensor_heads = np.zeros(n, np.int64)
        self.sensor_counts = np.zeros(n, np.int64)
        self.sensor_camera = np.zeros((n, 3), np.float64)
        self.ideal_obs = self.obs.copy()
        self.rewards = np.zeros(n, np.float32)
        self.done = np.zeros(n, np.bool_)
        self.reasons = np.zeros(n, np.int64)
        self.completed = []
        for e in range(n):
            self.reset_one(e)
        self.observe()

    def procedural(self):
        r = self.rng
        boxes = []
        large = []
        for k in range(self.tiles):
            by = 162 - 420 * k
            left = r.uniform(45, 200)
            boxes.extend([[1, by, left, 38], [left + 171, by, 248 - left, 38]])
            if self.stage >= 1 and k >= 5:
                for dy in (120, 291):
                    if r.random() < self.difficulty:
                        boxes.append([r.uniform(60, 330), by + dy, 30, 30])
            if self.stage >= 2 and k >= self.big_start and r.random() < 0.18:
                big_y = by - 240
                if r.random() < self.pair_probability:
                    large.extend([[72, big_y, 97, 97], [250, big_y, 97, 97]])
                else:
                    large.append([r.uniform(*self.large_x), big_y, 97, 97])
        boxes = [b for b in boxes if b[2] != 30 or all(abs(b[1] - g[1]) > 150 for g in large)]
        return np.array(boxes + large)

    def reset_one(self, e):
        self.boxes[e] = 0
        if self.courses is not None:
            course = self.courses[e % len(self.courses)]
            boxes = np.array(course["boxes"])
            x, y, vx, vy = course["initial"]
            height = course["height"]
        else:
            boxes = self.procedural()
            x, y, vx, vy, height = 209.5, 523.0, 0.0, 0.0, 736.0
        self.boxes[e, : len(boxes)] = boxes
        self.s[e] = 0
        self.done[e] = False
        self.reasons[e] = 0
        self.s[e, :5] = x, y, vx, vy, y
        self.s[e, 6] = 1.0
        self.s[e, 12] = y
        self.s[e, 14] = boxes[:, 1].min() - 35
        self.params[e] = 150.4, 688.55, 1415, height, 0.011, 0.04, 0.01, 0.025, self.processing
        if self.stage >= 3 and self.courses is None:
            r = self.rng
            self.params[e] = (
                r.uniform(145, 155),
                r.uniform(670, 710),
                r.uniform(1360, 1450),
                r.uniform(700, 780),
                r.uniform(0.002, 0.022),
                r.integers(6, 12) * DT,
                r.integers(0, 5) * DT,
                r.uniform(0.005, 0.045),
                r.uniform(0.003, 0.025),
            )
        self.heads[e] = 0
        self.history[e] = 0
        self.history[e, :, :6] = self.s[e, :6]
        self.sensor_samples[e] = 0
        self.sensor_heads[e] = 0
        self.sensor_counts[e] = 0
        self.sensor_camera[e] = 0
        if e < self.pixel_envs:
            from jumpbot.physics import Physics, Tracker
            from jumpbot.vision import Detector

            self.pixel_trackers[e] = Tracker(Physics(150.4, 688.55, 1415), reset_phase=0.0)
            self.pixel_detectors[e] = Detector((0, 0, 420, round(self.params[e, 3])))

    def observe(self):
        _observe(
            self.s,
            self.params,
            self.boxes,
            self.history,
            self.heads,
            self.obs,
            self.observation_version,
        )
        if self.observation_version >= 3:
            _tracked_globals(
                self.s,
                self.params,
                self.history,
                self.heads,
                self.sensor_samples,
                self.sensor_heads,
                self.sensor_counts,
                self.sensor_camera,
                self.obs,
            )
        self.ideal_obs[:] = self.obs
        if self.pixel_envs:
            from tools.offline_course_bench import render
            from tools.rl_policy import encode_state

            for e in range(self.pixel_envs):
                if self.s[e, 13] < 0:
                    continue
                height = round(self.params[e, 3])
                delay = round(self.params[e, 6] / DT)
                visual = self.history[e, (self.heads[e] - delay) % 32]
                x, y, _, _, peak, scene = visual[:6]
                boxes = self.boxes[e]
                visible = (
                    (boxes[:, 2] > 0)
                    & (boxes[:, 1] + scene < height)
                    & (boxes[:, 1] + boxes[:, 3] + scene > 0)
                )
                course = {"height": height, "boxes": boxes[visible]}
                target = max(0, height / 2 - peak)
                image = render(
                    course, (x, y, target, scene, int(max(0, 523 - peak) / 420)), *self.pixel_assets
                )
                state = self.pixel_trackers[e].update(
                    self.pixel_detectors[e].observe(image), self.s[e, 10]
                )
                if state is not None:
                    self.obs[e] = encode_state(
                        state, height, self.s[e, 8], self.observation_version
                    )
                    if not state.ready:
                        self.obs[e, 4] = 0
                else:
                    self.obs[e, 4] = 0
        return self.obs

    def action_risks(self, horizon=0.30):
        """Label only the committed prefix before another input could take effect.

        A wait can be followed by a jump at the next observation. A reset must
        also respect its new cooldown. Extending either label beyond that point
        would incorrectly punish actions that can be rescued by a later input.
        There is no rollout or extra model at policy inference time.
        """
        risks = np.zeros((self.n, 3), np.float32)
        params = self.params.copy()
        for action in range(3):
            period = self.params[:, 5]
            process = self.params[:, 8]
            lag = self.params[:, 7]
            if action:
                next_control = np.ceil((0.12 + process - 1e-9) / period) * period
            else:
                next_control = period
            params[:, 5] = np.minimum(horizon, next_control + process + lag)
            reasons = np.zeros(self.n, np.int64)
            _step(
                self.s.copy(),
                params,
                self.boxes,
                self.history.copy(),
                self.heads.copy(),
                np.full(self.n, action, np.int64),
                np.zeros(self.n, np.float32),
                np.zeros(self.n, np.bool_),
                reasons,
            )
            risks[:, action] = reasons == 1
        return risks

    def step(self, actions, auto_reset=True):
        was_done = self.done.copy()
        for e in range(self.pixel_envs):
            if self.s[e, 13] >= 0 and actions[e] and self.s[e, 6] >= 0.12 and self.s[e, 7] <= 0:
                expected = self.s[e, 10] + self.params[e, 8] + 0.035
                self.pixel_trackers[e].clicked(-1 if actions[e] == 1 else 1, expected)
        _step(
            self.s,
            self.params,
            self.boxes,
            self.history,
            self.heads,
            np.asarray(actions, np.int64),
            self.rewards,
            self.done,
            self.reasons,
        )
        rewards, done = self.rewards.copy(), self.done.copy()
        for e in np.flatnonzero(done if auto_reset else done & ~was_done):
            self.completed.append(
                {
                    "return": float(self.s[e, 15]),
                    "climb": float(523 - self.s[e, 4]),
                    "seconds": float(self.s[e, 10]),
                    "reason": int(self.reasons[e]),
                }
            )
            if auto_reset:
                self.reset_one(e)
            else:
                self.s[e, 13] = -1  # Keep terminal plants frozen during evaluation.
        return self.observe(), rewards, done
