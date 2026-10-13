"""Shared tremor feature extraction (used by training AND testing so they always match)."""
import numpy as np

SENSOR_COLUMNS = ["aX", "aY", "aZ", "gX", "gY", "gZ", "mX"]   # mY/mZ are constant (-1) in the dataset
WINDOW = 50      # samples per analysis window
STRIDE = 25      # 50% overlap
SEGMENT_LEN = 250  # samples per selectable "tremor dataset" (test case)


def window_features(w):
    """w: (WINDOW, 7) array -> 1D feature vector.
    Mean (posture/orientation) is intentionally excluded so the model learns
    movement/tremor, not how the sensor happened to be held."""
    w = np.asarray(w, dtype=float)
    mag = np.linalg.norm(w[:, :3], axis=1)
    feats = []
    for c in list(w.T) + [mag]:
        dc = c - c.mean()
        feats += [c.std(), np.abs(dc).mean(), c.max() - c.min(),
                  np.sqrt((dc ** 2).mean()), np.abs(np.diff(c)).mean(),
                  ((dc[:-1] * dc[1:]) < 0).mean()]
        sp = np.abs(np.fft.rfft(dc * np.hanning(len(dc)))) ** 2
        sp = sp[1:]
        total, n = sp.sum() + 1e-9, len(sp)
        for a, b in [(0, .1), (.1, .2), (.2, .35), (.35, .6), (.6, 1)]:
            lo = int(a * n)
            feats.append(sp[lo:max(int(b * n), lo + 1)].sum() / total)
        feats.append(np.argmax(sp) / n)
    return feats


def segment_windows(seg):
    """Yield features for every window inside one segment."""
    out = []
    for s in range(0, len(seg) - WINDOW + 1, STRIDE):
        out.append(window_features(seg[s:s + WINDOW]))
    return np.array(out)
