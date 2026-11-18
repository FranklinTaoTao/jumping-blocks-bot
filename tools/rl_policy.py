"""NumPy policy inference for offline benchmarks, with no Torch dependency."""

import numpy as np


class NumpyPolicy:
    def __init__(self, path):
        with np.load(path) as archive:
            self.observation_version = (
                int(archive["_observation_version"]) if "_observation_version" in archive else 1
            )
            self.weights = {k: archive[k].copy() for k in archive.files if not k.startswith("_")}

    def linear(self, x, name):
        return x @ self.weights[name + ".weight"].T + self.weights[name + ".bias"]

    def logits(self, obs):
        obs = np.asarray(obs, np.float32).reshape(-1, 92)
        objects = obs[:, 8:].reshape(-1, 12, 7)
        mask = objects[:, :, -1:]
        embedded = np.tanh(self.linear(objects, "objects.0"))
        embedded = np.tanh(self.linear(embedded, "objects.2"))
        count = mask.sum(1)
        mean = (embedded * mask).sum(1) / np.maximum(1, count)
        maximum = np.where(mask > 0, embedded, -1e9).max(1)
        maximum = np.where(count > 0, maximum, 0)
        body = np.concatenate([obs[:, :8], mean, maximum], axis=1)
        body = np.tanh(self.linear(body, "body.0"))
        body = np.tanh(self.linear(body, "body.2"))
        logits = self.linear(body, "actor")
        logits[obs[:, 4] < 0.999, 1:] = -1e9
        if self.observation_version >= 2:
            logits[(np.abs(obs[:, 5]) < 0.5) & (obs[:, 4] >= 0.999), 0] = -1e9
        return logits


def encode_state(state, height, direction=0, observation_version=2):
    """Match the training observation using only the detector/tracker's state."""
    obs = np.zeros(92, np.float32)
    vx = state.vx
    if (state.x < 1 and vx < 0) or (state.x > 419 and vx > 0):
        vx = 0
    obs[:5] = (
        state.x / 210 - 1,
        state.y / height * 2 - 1,
        vx / 150,
        state.vy / 700,
        np.clip((state.age_since_click + 0.035) / 0.12, 0, 1),
    )
    if observation_version >= 2:
        obs[5] = direction
    if state.age_since_click < 0:
        obs[5] = direction
        obs[6] = min(1, -state.age_since_click / 0.1)
    obs[7] = state.camera_speed / 700
    boxes = list(state.obstacles)
    if boxes:
        boxes = np.asarray(boxes)
        order = np.argsort(abs(boxes[:, 1] + boxes[:, 3] / 2 - state.y))[:12]
        for j, k in enumerate(order):
            bx, by, bw, bh, *velocity = boxes[k]
            vx, vy = (velocity + [0, 0])[:2]
            obs[8 + j * 7 : 8 + (j + 1) * 7] = (
                (bx + bw / 2 - state.x) / 210,
                (by + bh / 2 - state.y) / 420,
                bw / 210,
                bh / 420,
                vx / 150,
                vy / 700,
                1,
            )
    return obs


class RLPlanner:
    """Offline planner adapter, deliberately not registered with the live CLI."""

    def __init__(self, path):
        self.policy = NumpyPolicy(path)
        self.direction = 0

    def plan(self, state, height, latency=0.055, control_interval=0.04):
        from jumpbot.planner import Decision

        action = 0
        if state.ready:
            logits = self.policy.logits(
                encode_state(state, height, self.direction, self.policy.observation_version)
            )
            choice = int(logits.argmax())
            action = (0, -1, 1)[choice]
        if action:
            self.direction = action
        return Decision(action, np.empty((0, 2)), "offline learned policy")
