"""
stress_result.py

Turns one session row into a plain-language result for the GUI:

    * stress_percent  -- one 0-100 number (how stressed the person is)
    * factors         -- what is pushing that number up / down, found with
                         SHAP (explainable AI), in "percentage points"
    * advice          -- what to do about the factors that raise stress

How the percentage is computed
------------------------------
The models predict low / medium / high probabilities. We turn those into
one number with an expected-value score:

        stress % = 100 * (0.0*P(low) + 0.5*P(medium) + 1.0*P(high))

so 0% = confidently relaxed, 50% = medium, 100% = confidently high stress.

How the factors are computed
----------------------------
SHAP is run on THAT SAME score (not on one class), so every factor's value
reads directly as "this signal added +X points to the stress %" (positive,
red in the GUI) or "took X points off" (negative, green). The values add
up to (stress % - average % of the background data).

Each factor is also compared to the person's "relaxed baseline" -- the
median of the 'low' stress sessions in the training data -- so the text
can say "your heart rate was 96 bpm, relaxed sessions are around 67".

Model used: the TabNet feature-fusion model (same one the "Predict latest
session" button uses). If it hasn't been trained, falls back to the
decision-fusion (per-modality Random Forest) model.

Needs: pip install shap   (LIME is NOT required here)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import modality_utils as mu

BASE_DIR = Path(__file__).resolve().parent.parent

LEVEL_SCORE = {"low": 0.0, "medium": 0.5, "high": 1.0}

# ---------------------------------------------------------------------------
# Friendly names + advice.  key -> (label, unit, advice_if_above, advice_if_below)
# "above"/"below" = compared with the relaxed (low-stress) baseline.
# None means "no specific advice for that direction".
# ---------------------------------------------------------------------------
_BREATH = ("Try slow breathing for 3-5 minutes: breathe in for 4 seconds, "
           "out for 6. It lowers heart rate and calms the nervous system.")
_BREAK = "Take a proper break: step away from the screen, stretch and walk for 5-10 minutes."
_SLOW_TYPE = ("Slow down and type in short, relaxed bursts; relax your shoulders, wrists and "
              "fingers between sentences.")
_MOUSE_LOOSE = ("Loosen your grip on the mouse, rest your forearm on the desk and make slower, "
                "deliberate movements.")
_VOICE = ("Take a few deep breaths before speaking, speak slowly, and hum or gently stretch "
          "your neck and jaw to release throat tension.")
_FOCUS = ("Do one thing at a time. Close extra tabs and notifications and use a 25-minute focus "
          "block followed by a 5-minute rest.")

FEATURES = {
    # ---- ECG / heart ----
    "ecg_mean_hr_bpm": ("Average heart rate", "bpm", _BREATH + " Also drink water and cut back on caffeine.", None),
    "ecg_max_hr_bpm": ("Peak heart rate", "bpm", _BREATH, None),
    "ecg_min_hr_bpm": ("Lowest heart rate", "bpm", _BREATH, None),
    "ecg_mean_rr_ms": ("Time between heartbeats", "ms", None, _BREATH),
    "ecg_sdnn_ms": ("Heart-rate variability (SDNN)", "ms",
                    None, "Low heart-rate variability means your body is in 'alert' mode. Regular sleep, "
                          "light exercise and slow breathing raise it again."),
    "ecg_rmssd_ms": ("Heart-rate variability (RMSSD)", "ms",
                     None, "Low heart-rate variability is a sign of tension. Slow breathing, a walk in "
                           "daylight and good sleep help the body relax."),
    "ecg_pnn50_pct": ("Heart-rate variability (pNN50)", "%", None, _BREATH),
    "ecg_cvnn_pct": ("Heartbeat irregularity (CVNN)", "%", None, _BREATH),
    "ecg_r_peak_count": ("Heartbeats counted", "", _BREATH, None),
    # ---- keystroke ----
    "ks_words_per_minute": ("Typing speed", "wpm", None, _SLOW_TYPE.replace("Slow down and type", "Type") +
                            " Slower typing than usual often means the mind is overloaded - " + _FOCUS),
    "ks_backspace_count": ("Typing corrections (backspaces)", "", _SLOW_TYPE, None),
    "ks_avg_dwell_time_ms": ("Key press duration", "ms", None, "Very short, tense key presses: relax your hands, "
                             "shake them out and press keys lightly."),
    "ks_avg_flight_time_ms": ("Time between keys", "ms", None, _SLOW_TYPE),
    "ks_pause_count": ("Pauses while typing", "", _FOCUS, None),
    "ks_total_pause_time_sec": ("Total time paused while typing", "s", _FOCUS, None),
    "ks_avg_pause_duration_sec": ("Length of typing pauses", "s", _FOCUS, None),
    "ks_total_keystrokes": ("Number of keystrokes", "", None, None),
    "ks_correct_characters": ("Correctly typed characters", "", None, _SLOW_TYPE),
    "ks_total_typing_time_sec": ("Total typing time", "s", None, None),
    # ---- mouse ----
    "mouse_mouse_stress_score": ("Mouse stress score", "", _MOUSE_LOOSE, None),
    "mouse_mouse_avg_speed_px_s": ("Mouse speed", "px/s", _MOUSE_LOOSE, None),
    "mouse_mouse_max_speed_px_s": ("Peak mouse speed", "px/s", _MOUSE_LOOSE, None),
    "mouse_mouse_speed_std_px_s": ("Mouse speed variation", "px/s", _MOUSE_LOOSE, None),
    "mouse_mouse_avg_acceleration_px_s2": ("Mouse acceleration", "", _MOUSE_LOOSE, None),
    "mouse_mouse_avg_jerk_px_s3": ("Jerkiness of mouse movement", "", _MOUSE_LOOSE, None),
    "mouse_mouse_path_straightness_ratio": ("Straightness of mouse paths", "", None,
                                            "Your mouse paths are wandering. Slow down and aim deliberately."),
    "mouse_mouse_direction_change_count": ("Mouse direction changes", "", _MOUSE_LOOSE, None),
    "mouse_click_count": ("Mouse clicks", "", "Lots of rapid clicking can signal restlessness. " + _FOCUS, None),
    "mouse_left_click_count": ("Left clicks", "", "Lots of rapid clicking can signal restlessness. " + _FOCUS, None),
    "mouse_right_click_count": ("Right clicks", "", None, None),
    "mouse_mouse_avg_click_dwell_time_ms": ("Click hold time", "ms", None, _MOUSE_LOOSE),
    "mouse_mouse_avg_time_between_clicks_ms": ("Time between clicks", "ms", None,
                                               "Rapid-fire clicking: pause and take a breath between actions."),
    "mouse_mouse_scroll_count": ("Scrolling", "", None, None),
    "mouse_pause_count": ("Mouse pauses", "", _FOCUS, None),
    "mouse_total_pause_time_sec": ("Total mouse pause time", "s", _FOCUS, None),
    "mouse_avg_pause_duration_sec": ("Mouse pause length", "s", _FOCUS, None),
    "mouse_game_hit_rate": ("Target-game accuracy", "", None,
                            "Your accuracy dropped - a sign of reduced concentration. " + _BREAK),
    "mouse_game_hits": ("Targets hit", "", None, "Your accuracy dropped - a sign of reduced concentration. " + _BREAK),
    "mouse_game_misses": ("Targets missed", "", "You missed more targets than usual. " + _BREAK, None),
    "mouse_game_stray_clicks": ("Stray clicks", "", "Many stray clicks suggest tension. " + _MOUSE_LOOSE, None),
    "mouse_game_avg_reaction_time_ms": ("Reaction time", "ms", "Slower reactions suggest mental fatigue. " + _BREAK, None),
    "mouse_game_reaction_time_std_ms": ("Reaction-time consistency", "ms", "Your reactions were inconsistent. " + _FOCUS, None),
    "mouse_game_avg_click_offset_px": ("Click accuracy (distance off target)", "px", _MOUSE_LOOSE, None),
    # ---- voice ----
    "voice_f0_mean": ("Voice pitch", "Hz", "Your pitch is higher than when relaxed. " + _VOICE, None),
    "voice_f0_std": ("Voice pitch variation", "Hz", None, "Your voice sounds flatter/tenser than usual. " + _VOICE),
    "voice_f0_max": ("Highest voice pitch", "Hz", _VOICE, None),
    "voice_f0_min": ("Lowest voice pitch", "Hz", None, None),
    "voice_spectral_centroid_mean": ("Voice brightness / tension", "Hz", _VOICE, None),
    "voice_spectral_rolloff_mean": ("Voice high-frequency content", "Hz", _VOICE, None),
    "voice_energy_mean": ("Voice energy", "", _VOICE, None),
    "voice_total_energy": ("Total voice energy", "", _VOICE, None),
    "voice_rms_mean": ("Voice loudness", "", _VOICE, None),
    "voice_rms_max": ("Peak voice loudness", "", _VOICE, None),
    "voice_zcr_mean": ("Voice sharpness", "", _VOICE, None),
    "voice_onset_strength_mean": ("Speech abruptness", "", _VOICE, None),
    "voice_chroma_mean": ("Voice tone profile", "", _VOICE, None),
}

# Columns that are measurement plumbing, not something a person can act on.
# They still count in the model; they are just grouped under "technical signals".
TECHNICAL = {
    "ecg_sample_rate_hz", "ecg_total_samples", "ecg_expected_samples", "ecg_sequence_gaps",
    "ecg_largest_gap", "ecg_invalid_lines", "ecg_recording_duration_sec", "ecg_adc_mean",
    "ecg_adc_std", "ecg_adc_min", "ecg_adc_max", "ecg_adc_range", "voice_sample_rate",
    "voice_prompt_index", "voice_duration", "mouse_total_tracking_time_sec",
    "mouse_total_mouse_events", "mouse_total_distance_px", "mouse_game_targets_shown",
}

MODALITY_LABEL = {"keystroke": "Typing", "mouse": "Mouse", "voice": "Voice", "ecg": "Heart (ECG)"}

# (upper bound %, name, level used for advice, plain-language meaning)
BANDS = [
    (20, "Calm", "low",
     "You are calm and relaxed. Nothing in your typing, mouse, voice or heartbeat suggests stress."),
    (40, "Slightly tense", "low",
     "You show a little tension, which is normal during a busy day. You are not stressed in a way "
     "that needs treatment, but it is a good time to keep habits healthy."),
    (60, "Moderately stressed", "moderate",
     "You ARE showing clear signs of stress - roughly half-way between relaxed and very stressed. "
     "It is manageable, but worth acting on today before it builds up."),
    (80, "Highly stressed", "high",
     "You ARE stressed. Your body and behaviour look like someone under heavy pressure, well above "
     "your relaxed state. Please slow down and use the steps below now."),
    (101, "Severely stressed", "high",
     "You are very strongly stressed right now - almost everything we measure points the same way. "
     "Stop and take care of yourself first; the steps below start with what to do in the next 10 minutes."),
]

# Each level gets a different plan (sections of tips).  "{factors}" marks where the
# tips tailored to this person's own causes are inserted.
LEVEL_PLAN = {
    "low": [
        ("Keep it this way", [
            "Keep your routine: 7-8 hours of sleep, regular meals and some daily movement (a 20-30 minute walk is enough).",
            "Keep taking short breaks - 5 minutes every hour - even when you feel fine.",
            "Drink water through the day and keep caffeine moderate.",
            "Spend a little time on something you enjoy today (music, a call with a friend, a hobby).",
        ]),
        ("{factors}", []),
        ("Next check", [
            "No action needed. Run this again if your workload, deadlines or sleep change.",
        ]),
    ],
    "moderate": [
        ("Do this today", [
            "Take a 5-minute breathing break now: breathe in for 4 seconds, out for 6, repeat 10 times.",
            "Choose your top 3 tasks for today and let the rest wait - an overloaded list adds stress.",
            "Take a break every 45-60 minutes: stand up, stretch, look away from the screen.",
            "Go for a 10-15 minute walk, ideally outside, and drink a glass of water.",
            "Cut back on caffeine for the rest of the day.",
        ]),
        ("{factors}", []),
        ("This week", [
            "Aim for a regular sleep time and switch screens off 30 minutes before bed.",
            "If you feel like this most days, talk to a friend, a mentor or a counsellor about what is weighing on you.",
        ]),
    ],
    "high": [
        ("Right now (next 10 minutes)", [
            "Stop what you are doing. Sit back, drop your shoulders and unclench your jaw and hands.",
            "Slow breathing for 3-5 minutes: in through the nose for 4 seconds, out slowly for 6-8 seconds.",
            "Drink a glass of water and step away from the screen - walk or stretch somewhere quiet.",
            "Do not make big decisions or send heated messages until you feel calmer.",
        ]),
        ("Later today", [
            "Cut your to-do list down to what is truly urgent and postpone the rest.",
            "Skip extra caffeine, and have a proper meal.",
            "Talk to someone you trust about how you feel - saying it out loud lowers the pressure.",
            "Go to bed early tonight and keep screens away for the last 30 minutes.",
        ]),
        ("{factors}", []),
        ("Get support if it continues", [
            "If you feel this stressed for several days, or you have trouble sleeping, chest tightness, panic or "
            "feel unable to cope, please speak to a doctor or counsellor.",
            "In India, Tele-MANAS (14416, free, 24x7) offers mental-health support. If you ever feel unsafe, "
            "contact local emergency services or someone close to you right away.",
        ]),
    ],
}

FACTOR_SECTION_TITLE = {
    "low": "Small things to keep an eye on",
    "moderate": "Based on what we detected in YOU",
    "high": "Based on what we detected in YOU",
}
FACTOR_SECTION_MAX = {"low": 2, "moderate": 4, "high": 5}


def band_for_percent(p: float):
    for upper, name, level, meaning in BANDS:
        if p < upper:
            return name, level, meaning
    return BANDS[-1][1:4]


def level_from_percent(p: float) -> str:
    return band_for_percent(p)[1]


def _pretty(col: str) -> str:
    base, mod = col, ""
    for pre, name in (("ks_", "Typing"), ("mouse_", "Mouse"), ("voice_", "Voice"), ("ecg_", "Heart")):
        if base.startswith(pre):
            base, mod = base[len(pre):], name
            break
    if base.startswith("mfcc_"):
        return "Voice tone"
    suffix = ""
    if base.endswith("_std"):
        base, suffix = base[:-4], " variation"
    elif base.endswith("_mean"):
        base = base[:-5]
    text = base.replace("_", " ").strip()
    if text.startswith(("mouse ", "game ")):
        text = text.split(" ", 1)[1] if text.startswith("mouse ") else text
    text = text.replace("px s", "").replace("ms", "").strip()
    if mod and not text.lower().startswith(mod.lower()):
        text = f"{mod.lower()} {text}"
    return (text + suffix).strip().capitalize()


MODALITY_ADVICE = {
    "keystroke": _SLOW_TYPE,
    "mouse": _MOUSE_LOOSE,
    "voice": _VOICE,
    "ecg": _BREATH,
}


def _fmt(v, unit):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if abs(v) >= 100:
        s = f"{v:,.0f}"
    elif abs(v) >= 10:
        s = f"{v:.1f}"
    elif abs(v) >= 1:
        s = f"{v:.2f}"
    else:
        s = f"{v:.3g}"
    return f"{s} {unit}".strip()


# ---------------------------------------------------------------------------
# Model access: returns (predict_proba(X)->(n,classes), feature_cols, classes, kind)
# ---------------------------------------------------------------------------
def _load_scorer():
    import predict as pr
    import decision_fusion as dfm

    if (pr.TABNET_ARTIFACTS_DIR / "stress_model.zip").exists():
        model, feature_cols, classes = pr.load_model(pr.TABNET_ARTIFACTS_DIR)
        return model.predict_proba, feature_cols, classes, "TabNet (feature-level fusion)"

    if (dfm.ARTIFACTS_DIR / "metrics.json").exists():
        submodels, weights, classes = dfm.load_submodels(dfm.ARTIFACTS_DIR)
        feature_cols = []
        for sm in submodels.values():
            feature_cols += [c for c in sm["feature_columns"] if c not in feature_cols]
        idx = {c: i for i, c in enumerate(feature_cols)}
        total_w = sum(weights.get(m, 0.0) for m in submodels) or 1.0

        def predict_proba(X):
            out = np.zeros((len(X), len(classes)))
            for m, sm in submodels.items():
                cols = [idx[c] for c in sm["feature_columns"]]
                p = sm["pipeline"].predict_proba(X[:, cols])
                for local, glob in enumerate(sm["pipeline"].named_steps["clf"].classes_):
                    out[:, int(glob)] += weights.get(m, 0.0) / total_w * p[:, local]
            return out

        return predict_proba, feature_cols, classes, "Decision-level fusion (per-modality Random Forests)"

    raise RuntimeError("No trained model found. Click 'Rebuild merged dataset' and then train "
                       "'feature fusion (TabNet)' first.")


def compute_result(row, nsamples: int = 400) -> dict:
    """row: a merged-dataset-style pandas Series (e.g. predict.get_latest_row())."""
    import shap
    import explain as ex

    predict_proba, feature_cols, classes, model_name = _load_scorer()
    background = mu.load_merged()

    # A model trained on only one stress level can only ever answer that level
    # (e.g. only 'low' rows -> always 0%).  Refuse instead of showing a misleading number.
    counts = background["stress_level"].astype(str).str.lower().value_counts().to_dict()
    if len(classes) < 2 or len(counts) < 2:
        have = ", ".join(f"{k}: {v}" for k, v in counts.items()) or "none"
        raise RuntimeError(
            "The model was trained on only ONE stress level, so it would always give the same answer "
            f"(your merged dataset has -> {have}).\n\n"
            "Fix: collect/generate sessions for low, medium AND high stress (e.g. run "
            "create_synthetic.py --merge), then click 'Rebuild merged dataset' and "
            "'Train feature fusion (TabNet)' again.")
    weights = np.array([LEVEL_SCORE.get(str(c).lower(), 0.5) for c in classes])

    def score(X):  # 0..100 stress percent
        return predict_proba(np.asarray(X, dtype=float)) @ weights * 100.0

    x = ex._fill_row(row, background, feature_cols).reshape(1, -1)
    proba = predict_proba(x)[0]
    percent = float(score(x)[0])

    bg = ex._background_matrix(background, feature_cols)
    explainer = shap.KernelExplainer(score, bg)
    sv = np.asarray(explainer.shap_values(x, nsamples=nsamples, silent=True)).reshape(-1)
    base_value = float(np.ravel(explainer.expected_value)[0])

    # relaxed baseline = median of 'low' sessions
    low_rows = background[background["stress_level"].astype(str).str.lower() == "low"]
    baseline = (low_rows if len(low_rows) else background)[feature_cols].median(numeric_only=True)

    factors, technical_total = [], 0.0
    for col, val, shap_val in zip(feature_cols, x[0], sv):
        if col in TECHNICAL:
            technical_total += float(shap_val)
            continue
        label, unit, adv_hi, adv_lo = FEATURES.get(col, (_pretty(col), "", None, None))
        base = float(baseline.get(col, np.nan))
        above = bool(val > base)
        factors.append({
            "feature": col,
            "label": label,
            "modality": mu.modality_of_column(col) or "other",
            "value": float(val),
            "value_text": _fmt(float(val), unit),
            "relaxed_text": _fmt(base, unit),
            "above_relaxed": above,
            "impact": float(shap_val),      # percentage points
            "advice": ((adv_hi if above else adv_lo) if col in FEATURES
                       else MODALITY_ADVICE.get(mu.modality_of_column(col))) if shap_val > 0 else None,
        })

    factors.sort(key=lambda f: -abs(f["impact"]))
    raising = [f for f in factors if f["impact"] > 0.05]
    lowering = [f for f in factors if f["impact"] < -0.05]

    by_mod = {}
    for f in factors:
        by_mod[f["modality"]] = by_mod.get(f["modality"], 0.0) + f["impact"]
    modality_impact = {MODALITY_LABEL.get(m, m): v for m, v in
                       sorted(by_mod.items(), key=lambda t: -abs(t[1]))}

    trained_on = None
    try:
        import predict as _pr, decision_fusion as _df
        mdir = _pr.TABNET_ARTIFACTS_DIR if model_name.startswith("TabNet") else _df.ARTIFACTS_DIR
        trained_on = json.loads((mdir / "metrics.json").read_text()).get("n_samples")
    except Exception:
        pass

    level = level_from_percent(percent)
    summary = plain_summary(percent, raising, lowering)
    return {
        "band": summary["band"], "headline": summary["headline"],
        "meaning": summary["meaning"], "reasons": summary["reasons"],
        "session_id": str(row.get("session_id", "")),
        "participant_id": str(row.get("participant_id", "")),
        "model": model_name,
        "trained_on": trained_on,
        "stress_percent": percent,
        "level": level,
        "probabilities": {str(c): float(p) for c, p in zip(classes, proba)},
        "average_percent": base_value,
        "raising": raising[:8],
        "lowering": lowering[:4],
        "modality_impact": modality_impact,
        "technical_impact": technical_total,
        "advice": build_advice(level, raising),
        "short_advice": build_short_advice(level, raising),
        "found_features_missing": int(sum(pd.isna(row.get(c, np.nan)) for c in feature_cols)),
    }


def build_advice(level: str, raising: list) -> list:
    """Returns [{"title": str, "tips": [str, ...]}, ...] -- a different plan for each
    stress level, with tips tailored to this person's own top causes inserted."""
    factor_tips, seen = [], set()
    for f in raising:
        a = f.get("advice")
        if a and a not in seen:
            seen.add(a)
            factor_tips.append(f"{f['label']}: {a}")
        if len(factor_tips) >= FACTOR_SECTION_MAX[level]:
            break

    sections = []
    for title, tips in LEVEL_PLAN[level]:
        if title == "{factors}":
            if factor_tips:
                sections.append({"title": FACTOR_SECTION_TITLE[level], "tips": factor_tips})
            elif level != "low":
                sections.append({"title": FACTOR_SECTION_TITLE[level], "tips": [
                    "No single signal stands out - the steps above (rest, water, a short walk, "
                    "fewer distractions) are the best place to start."]})
        else:
            sections.append({"title": title, "tips": list(tips)})
    return sections


SHORT_PLAN = {
    "low": ["Keep your routine: good sleep and short breaks",
            "Drink water and stay active"],
    "moderate": ["Take a 5-minute break and breathe slowly",
                 "Walk for 10 minutes and drink water",
                 "Focus on your top 3 tasks only"],
    "high": ["Stop and breathe slowly for 3-5 minutes",
             "Step away from the screen and drink water",
             "Postpone big decisions; talk to someone you trust"],
}
SHORT_BY_MODALITY = {
    "keystroke": "Typing: relax your hands and slow down",
    "mouse": "Mouse: loosen your grip and move slower",
    "voice": "Voice: breathe deeply and speak slowly",
    "ecg": "Heart: breathe in for 4 sec, out for 6 sec",
}
SHORT_SUPPORT = "Stressed for days? Talk to a doctor or counsellor (India: Tele-MANAS 14416)"


def build_short_advice(level: str, raising: list) -> list:
    """3-5 one-line tips: the level's plan + one line for each of the top signals."""
    tips = list(SHORT_PLAN[level])
    if level != "low":
        mods = []
        for f in raising:
            if f["modality"] in SHORT_BY_MODALITY and f["modality"] not in mods:
                mods.append(f["modality"])
            if len(mods) >= (1 if level == "high" else 2):
                break
        tips += [SHORT_BY_MODALITY[m] for m in mods]
    if level == "high":
        tips.append(SHORT_SUPPORT)
    return tips


def plain_summary(percent: float, raising: list, lowering: list) -> dict:
    """Plain-English explanation of the number, for people who don't read SHAP."""
    band, level, meaning = band_for_percent(percent)
    total = sum(f["impact"] for f in raising) or 1.0
    reasons = []
    for f in raising[:4]:
        share = f["impact"] / total
        f["share"] = share
        f["importance"] = "Major cause" if share >= 0.30 else ("Moderate cause" if share >= 0.12 else "Minor cause")
        f["plain"] = (f"{f['label']} is {'higher' if f['above_relaxed'] else 'lower'} than when you are relaxed "
                      f"({f['value_text']} now vs {f['relaxed_text']} normally)")
        reasons.append(f["plain"])
    for f in raising[4:]:
        share = f["impact"] / total
        f["share"] = share
        f["importance"] = "Moderate cause" if share >= 0.12 else "Minor cause"
        f["plain"] = (f"{f['label']} is {'higher' if f['above_relaxed'] else 'lower'} than when you are relaxed "
                      f"({f['value_text']} now vs {f['relaxed_text']} normally)")
    for f in lowering:
        f["plain"] = (f"{f['label']} is {'higher' if f['above_relaxed'] else 'lower'} than when you are relaxed "
                      f"({f['value_text']} now vs {f['relaxed_text']} normally) - this is helping you")
    if level == "low":
        headline = f"You are NOT stressed - {band.lower()}"
    elif level == "moderate":
        headline = "You are moderately stressed"
    else:
        headline = f"You are stressed - {band.lower()}"
    return {"band": band, "headline": headline, "meaning": meaning, "reasons": reasons}


# ---------------------------------------------------------------------------
# QUICK ESTIMATE -- works with NO trained model.
# Each signal has a "relaxed" and a "stressed" reference value. The latest session's value
# is placed between them (0 = relaxed, 1 = stressed), and the average over the available
# signals is the stress percent. If your merged dataset has both low and high sessions, their
# medians replace these built-in (generic, literature-style) references automatically.
# ---------------------------------------------------------------------------
DEFAULT_REFS = {
    "ecg_mean_hr_bpm": (66.89, 95.2),
    "ecg_rmssd_ms": (47.17, 18.93),
    "ecg_sdnn_ms": (55.86, 25.71),
    "ecg_pnn50_pct": (28.08, 6.335),
    "ks_words_per_minute": (58.39, 39.24),
    "ks_avg_dwell_time_ms": (118.8, 86.22),
    "ks_backspace_count": (5.0, 24.5),
    "ks_pause_count": (1.0, 6.0),
    "ks_avg_flight_time_ms": (117.0, 83.19),
    "mouse_game_hit_rate": (0.8395, 0.5115),
    "mouse_game_avg_reaction_time_ms": (517.5, 816.0),
    "mouse_game_reaction_time_std_ms": (86.95, 198.6),
    "mouse_mouse_stress_score": (0.2215, 0.6905),
    "mouse_mouse_avg_speed_px_s": (529.2, 824.5),
    "mouse_mouse_avg_jerk_px_s3": (323500.0, 660300.0),
    "mouse_mouse_path_straightness_ratio": (1.635, 2.553),
    "mouse_game_avg_click_offset_px": (8.25, 18.2),
    "voice_f0_mean": (165.4, 207.6),
    "voice_f0_std": (20.23, 40.62),
    "voice_spectral_centroid_mean": (1800.0, 2402.0),
    "voice_rms_mean": (0.08801, 0.1478),
    "voice_onset_strength_mean": (1.232, 1.957),
    "voice_zcr_mean": (0.06057, 0.0987),
}


def _reference_values(background=None) -> dict:
    refs = dict(DEFAULT_REFS)
    try:
        bg = background if background is not None else mu.load_merged()
        lvl = bg["stress_level"].astype(str).str.lower()
        lo, hi = bg[lvl == "low"], bg[lvl == "high"]
        if len(lo) >= 3 and len(hi) >= 3:
            for col in list(refs):
                if col in bg.columns:
                    l, h = lo[col].median(), hi[col].median()
                    sd = float(np.sqrt((lo[col].var() + hi[col].var()) / 2))
                    if pd.notna(l) and pd.notna(h) and sd > 0 and abs(h - l) / sd >= 0.5:
                        refs[col] = (float(l), float(h))
    except BaseException:
        pass
    return refs


def quick_estimate(row, note: str = "") -> dict:
    refs = _reference_values()
    scored = []                                    # (col, value, relaxed, stressed, s 0..1)
    for col, (lo, hi) in refs.items():
        try:
            v = float(row.get(col, np.nan))
        except (TypeError, ValueError):
            continue
        if np.isnan(v) or lo == hi:
            continue
        scored.append((col, v, lo, hi, float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))))
    if not scored:
        raise RuntimeError("The latest session has none of the measurements needed for an estimate "
                           "(typing, mouse, voice or heart). Record a session first.")

    # each modality counts equally, however many signals it has
    mods = {}
    for item in scored:
        mods.setdefault(mu.modality_of_column(item[0]) or "other", []).append(item)
    factors, percent = [], 0.0
    for m, items in mods.items():
        for col, v, lo, hi, s in items:
            w = 1.0 / (len(mods) * len(items))
            percent += s * w * 100.0
            label, unit, adv_hi, adv_lo = FEATURES.get(col, (_pretty(col), "", None, None))
            above = v > lo
            impact = (s - 0.5) * 100.0 * w
            factors.append({
                "feature": col, "label": label, "modality": m, "value": v,
                "value_text": _fmt(v, unit), "relaxed_text": _fmt(lo, unit),
                "above_relaxed": above, "impact": impact,
                "advice": (adv_hi if above else adv_lo) if impact > 0 else None,
            })
    factors.sort(key=lambda f: -abs(f["impact"]))
    raising = [f for f in factors if f["impact"] > 0.05]
    lowering = [f for f in factors if f["impact"] < -0.05]
    by_mod = {}
    for f in factors:
        by_mod[f["modality"]] = by_mod.get(f["modality"], 0.0) + f["impact"]
    modality_impact = {MODALITY_LABEL.get(m, m): v for m, v in
                       sorted(by_mod.items(), key=lambda t: -abs(t[1]))}

    level = level_from_percent(percent)
    summary = plain_summary(percent, raising, lowering)
    return {
        "band": summary["band"], "headline": summary["headline"],
        "meaning": summary["meaning"], "reasons": summary["reasons"],
        "session_id": str(row.get("session_id", "")),
        "participant_id": str(row.get("participant_id", "")),
        "model": "Quick estimate (no trained model needed)",
        "note": note or "Quick estimate from your latest session's typing, mouse, voice and heart signals.",
        "trained_on": None,
        "stress_percent": percent, "level": level, "probabilities": {},
        "average_percent": 50.0,
        "raising": raising[:8], "lowering": lowering[:4],
        "modality_impact": modality_impact, "technical_impact": 0.0,
        "advice": build_advice(level, raising),
        "short_advice": build_short_advice(level, raising),
        "found_features_missing": 0,
    }


def latest_result(force_quick: bool = False) -> dict:
    """Stress result for the latest session. Uses the trained model + SHAP when it is usable;
    otherwise (not trained, only one stress level in the data, any failure) falls back to the
    quick estimate so the button always gives an answer."""
    import predict as pr
    row = pr.get_latest_row()
    if force_quick:
        return quick_estimate(row)
    try:
        return compute_result(row)
    except (Exception, SystemExit) as e:
        first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        return quick_estimate(row, note=f"Quick estimate - the trained model could not be used ({first})")


if __name__ == "__main__":
    print(json.dumps(latest_result(), indent=2))
