"""
result_window.py

Tkinter pop-up that shows a stress result (from stress_result.compute_result):
  * the stress percentage + a coloured gauge
  * the factors raising / lowering it (explainable AI / SHAP), as bars
  * advice on what to do

Uses tkinter only, so it can be tested on its own:
    python result_window.py result.json
"""

import json
import sys
import tkinter as tk
from tkinter import ttk

BG, PANEL, FG, MUTED = "#1b2226", "#222b2f", "#e7ede8", "#8a9a96"
GREEN, AMBER, RED, BLUE = "#3fb950", "#e3b341", "#f85149", "#3a8fd9"

LEVEL_STYLE = {
    "low": ("CALM", GREEN),
    "moderate": ("MODERATELY STRESSED", AMBER),
    "high": ("HIGHLY STRESSED", RED),
}


def _lerp(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _gauge_color(frac):
    return _lerp(GREEN, AMBER, frac / 0.5) if frac < 0.5 else _lerp(AMBER, RED, (frac - 0.5) / 0.5)


class DetailsWindow(tk.Toplevel):
    def __init__(self, master, result: dict):
        super().__init__(master)
        self.result = result
        self.title("Stress Result - full details")
        self.geometry("780x860")
        self.minsize(640, 520)
        self.configure(bg=BG)

        # --- scrollable body -----------------------------------------------
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        vbar = ttk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.body = tk.Frame(self.canvas, bg=BG)
        win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(win, width=e.width))
        self.bind_all("<MouseWheel>", self._wheel)
        self.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-2, "units"))
        self.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(2, "units"))

        self._build(result)

    def _wheel(self, e):
        self.canvas.yview_scroll(int(-e.delta / 60) or (-1 if e.delta > 0 else 1), "units")

    # -- helpers ---------------------------------------------------------
    def _card(self, title):
        box = tk.LabelFrame(self.body, text=f" {title} ", bg=PANEL, fg=FG, padx=14, pady=10,
                            font=("Helvetica", 11, "bold"), bd=1, relief="solid")
        box.pack(fill="x", padx=18, pady=(0, 12))
        return box

    def _para(self, parent, text, fg=FG, size=10, bold=False, pady=(0, 0), wrap=690):
        tk.Label(parent, text=text, bg=parent["bg"], fg=fg, justify="left", anchor="w", wraplength=wrap,
                 font=("Helvetica", size, "bold" if bold else "normal")).pack(fill="x", pady=pady)

    def _build(self, r):
        pct = max(0.0, min(100.0, r["stress_percent"]))
        color = LEVEL_STYLE[r["level"]][1]
        band = r.get("band", LEVEL_STYLE[r["level"]][0]).upper()

        tk.Label(self.body, text="Your Stress Result", bg=BG, fg=FG,
                 font=("Helvetica", 18, "bold")).pack(pady=(16, 0))
        sub = " | ".join(x for x in (f"Participant: {r.get('participant_id')}" if r.get("participant_id") else "",
                                      f"Session: {r.get('session_id')}" if r.get("session_id") else "") if x)
        if sub:
            tk.Label(self.body, text=sub, bg=BG, fg=MUTED, font=("Helvetica", 9)).pack()

        # --- 1. percentage, verdict, gauge -----------------------------------
        top = self._card("Your stress level")
        tk.Label(top, text=f"{pct:.0f}%", bg=PANEL, fg=color, font=("Helvetica", 48, "bold")).pack()
        tk.Label(top, text="of the way from completely relaxed (0%) to extremely stressed (100%)",
                 bg=PANEL, fg=MUTED, font=("Helvetica", 9)).pack()
        tk.Label(top, text=r.get("headline", band), bg=PANEL, fg=color,
                 font=("Helvetica", 15, "bold"), wraplength=680).pack(pady=(8, 4))
        if r.get("meaning"):
            tk.Label(top, text=r["meaning"], bg=PANEL, fg=FG, justify="center", wraplength=640,
                     font=("Helvetica", 10)).pack(pady=(0, 8))

        gw, gh = 640, 26
        g = tk.Canvas(top, width=gw, height=gh + 40, bg=PANEL, highlightthickness=0)
        g.pack(pady=(4, 2))
        steps = 64
        for i in range(steps):
            g.create_rectangle(i * gw / steps, 0, (i + 1) * gw / steps + 1, gh,
                               fill=_gauge_color(i / (steps - 1)), outline="")
        for edge in (20, 40, 60, 80):
            g.create_line(edge / 100 * gw, 0, edge / 100 * gw, gh, fill=PANEL, width=2)
        x = min(max(pct / 100 * gw, 4), gw - 4)
        g.create_polygon(x - 8, gh + 18, x + 8, gh + 18, x, gh + 3, fill=FG, outline="")
        g.create_line(x, 0, x, gh, fill=FG, width=3)
        for lo, hi, name in ((0, 20, "Calm"), (20, 40, "Slightly tense"), (40, 60, "Moderate"),
                              (60, 80, "High"), (80, 100, "Severe")):
            g.create_text((lo + hi) / 2 / 100 * gw, gh + 30, text=name, fill=MUTED, font=("Helvetica", 8))

        probs = r.get("probabilities", {})
        if probs:
            order = [k for k in ("low", "medium", "high") if k in probs]
            tk.Label(top, text="Model confidence:  " + "   ".join(f"{k} {probs[k]:.0%}" for k in order),
                     bg=PANEL, fg=MUTED, font=("Helvetica", 9)).pack(pady=(6, 0))

        # --- 2. why, in plain words -------------------------------------------
        raising = r.get("raising", [])
        lowering = r.get("lowering", [])
        if r["level"] == "low":
            why = self._card("How your body and behaviour look")
            self._para(why, "Your typing, mouse use, voice and heartbeat are close to what they look like when you "
                            "are relaxed.")
        else:
            why = self._card("Why we think you are stressed (in plain words)")
            for reason in r.get("reasons", []):
                self._para(why, "\u2022  " + reason, pady=(2, 2))
            if not r.get("reasons"):
                self._para(why, "No single signal stands out; it is a combination of many small changes.")

        # --- 3. detailed causes (explainable AI) --------------------------------
        scale = max([abs(f["impact"]) for f in raising + lowering] + [1e-6])
        if raising:
            cause = self._card("What is causing it  (explainable AI / SHAP)")
            self._para(cause, "Each bar shows how much of your stress score a signal is responsible for. "
                              "Longer bar = bigger cause. Compared with your relaxed baseline.", fg=MUTED, size=9)
            for f in raising:
                self._factor_row(cause, f, scale, RED)

        if lowering:
            helping = self._card("What is keeping it down")
            for f in lowering:
                self._factor_row(helping, f, scale, GREEN)

        # --- 4. which sensors -----------------------------------------------------
        mods = r.get("modality_impact", {})
        pos_total = sum(v for v in mods.values() if v > 0)
        if mods and pos_total > 0:
            m = self._card("Where the stress shows up")
            for name, v in mods.items():
                if v > 0:
                    self._bar_row(m, name, "", v, max(mods.values()), RED, f"{v / pos_total:.0%} of the stress",
                                  label_w=110)
                else:
                    self._bar_row(m, name, "", v, max(mods.values()), GREEN, "calm", label_w=110)

        # --- 5. advice, specific to this level ----------------------------------------
        titles = {"low": "What to do  (you are doing well)", "moderate": "What to do  (act on it today)",
                  "high": "What to do  (please act on this now)"}
        adv = self._card(titles[r["level"]])
        n = 0
        for sec in r.get("advice", []):
            tk.Label(adv, text=sec["title"], bg=PANEL, fg=color, font=("Helvetica", 11, "bold"),
                     anchor="w").pack(fill="x", pady=(8, 2))
            for tip in sec["tips"]:
                n += 1
                row = tk.Frame(adv, bg=PANEL)
                row.pack(fill="x", pady=2)
                tk.Label(row, text=f"{n}.", bg=PANEL, fg=BLUE, font=("Helvetica", 10, "bold"),
                         width=3, anchor="ne").pack(side="left", anchor="n")
                tk.Label(row, text=tip, bg=PANEL, fg=FG, justify="left", anchor="w", wraplength=650,
                         font=("Helvetica", 10)).pack(side="left", fill="x", expand=True)

        tk.Label(self.body, justify="center", wraplength=700, bg=BG, fg=MUTED, font=("Helvetica", 8),
                 text="This is an automated estimate from typing, mouse, voice and heart signals. "
                      "It is not a medical diagnosis.").pack(pady=(0, 6))
        tk.Button(self.body, text="Close", bg=BLUE, fg="white", activebackground=BLUE, width=14,
                  command=self.destroy).pack(pady=(0, 18))

    def _factor_row(self, parent, f, scale, color):
        tag = f.get("importance")
        if f["impact"] < 0:
            tag = "Helping"
        elif tag and f.get("share") is not None:
            tag = f"{tag} ({f['share']:.0%})"
        self._bar_row(parent, f["label"], f.get("plain", ""), f["impact"], scale, color, tag or "")

    def _bar_row(self, parent, label, detail, value, scale, color, tag, label_w=230):
        box = tk.Frame(parent, bg=PANEL)
        box.pack(fill="x", pady=4)
        row = tk.Frame(box, bg=PANEL)
        row.pack(fill="x")
        tk.Label(row, text=label, bg=PANEL, fg=FG, anchor="w", width=max(1, label_w // 8),
                 font=("Helvetica", 10, "bold")).pack(side="left")
        bw = 260
        c = tk.Canvas(row, width=bw, height=16, bg=PANEL, highlightthickness=0)
        c.pack(side="left", padx=8)
        c.create_rectangle(0, 3, bw, 13, fill="#2f3b3f", outline="")
        c.create_rectangle(0, 3, max(3, abs(value) / scale * bw), 13, fill=color, outline="")
        tk.Label(row, text=tag, bg=PANEL, fg=color, anchor="w", font=("Helvetica", 9, "bold")).pack(side="left")
        if detail:
            tk.Label(box, text=detail, bg=PANEL, fg=MUTED, justify="left", anchor="w", wraplength=690,
                     font=("Helvetica", 9)).pack(fill="x")


def show_details(master, result: dict) -> "DetailsWindow":
    w = DetailsWindow(master, result)
    w.lift()
    w.focus_force()
    return w


SHORT_TAG = {"Major cause": "Major", "Moderate cause": "Moderate", "Minor cause": "Minor"}


class ResultWindow(tk.Toplevel):
    """Compact, glanceable result: percentage + verdict, top causes, a few short tips.
    The long explanation lives behind the 'Full details' button."""

    def __init__(self, master, result: dict):
        super().__init__(master)
        r = self.result = result
        self.title("Stress Result")
        self.configure(bg=BG)
        self.resizable(False, False)

        pct = max(0.0, min(100.0, r["stress_percent"]))
        color = LEVEL_STYLE[r["level"]][1]

        # --- headline: percentage + verdict + gauge -------------------------
        head = tk.Frame(self, bg=PANEL, padx=18, pady=12)
        head.pack(fill="x", padx=14, pady=(14, 8))
        tk.Label(head, text=f"{pct:.0f}%", bg=PANEL, fg=color, font=("Helvetica", 44, "bold")).pack()
        tk.Label(head, text=r.get("headline", ""), bg=PANEL, fg=color, font=("Helvetica", 13, "bold"),
                 wraplength=520).pack()
        gw, gh = 520, 16
        g = tk.Canvas(head, width=gw, height=gh + 20, bg=PANEL, highlightthickness=0)
        g.pack(pady=(10, 0))
        steps = 52
        for i in range(steps):
            g.create_rectangle(i * gw / steps, 0, (i + 1) * gw / steps + 1, gh,
                               fill=_gauge_color(i / (steps - 1)), outline="")
        x = min(max(pct / 100 * gw, 4), gw - 4)
        g.create_line(x, 0, x, gh, fill=FG, width=3)
        g.create_polygon(x - 6, gh + 12, x + 6, gh + 12, x, gh + 2, fill=FG, outline="")
        g.create_text(0, gh + 14, text="0% relaxed", fill=MUTED, font=("Helvetica", 8), anchor="w")
        g.create_text(gw, gh + 14, text="100% very stressed", fill=MUTED, font=("Helvetica", 8), anchor="e")

        # --- top causes (max 3) ---------------------------------------------------
        raising = r.get("raising", [])[:3]
        box = tk.Frame(self, bg=PANEL, padx=18, pady=10)
        box.pack(fill="x", padx=14, pady=4)
        tk.Label(box, text="MAIN CAUSES" if raising else "HOW YOU LOOK", bg=PANEL, fg=MUTED,
                 font=("Helvetica", 9, "bold"), anchor="w").pack(fill="x")
        if raising:
            scale = max(abs(f["impact"]) for f in raising)
            for f in raising:
                row = tk.Frame(box, bg=PANEL)
                row.pack(fill="x", pady=3)
                word = "higher" if f["above_relaxed"] else "lower"
                tk.Label(row, text=f"{f['label']} - {word} than usual", bg=PANEL, fg=FG, anchor="w",
                         font=("Helvetica", 10), width=46).pack(side="left")
                c = tk.Canvas(row, width=90, height=10, bg=PANEL, highlightthickness=0)
                c.pack(side="left", padx=6)
                c.create_rectangle(0, 0, 90, 10, fill="#2f3b3f", outline="")
                c.create_rectangle(0, 0, max(4, f["impact"] / scale * 90), 10, fill=RED, outline="")
                tk.Label(row, text=SHORT_TAG.get(f.get("importance", ""), ""), bg=PANEL, fg=RED,
                         font=("Helvetica", 9, "bold"), width=8, anchor="w").pack(side="left")
        else:
            tk.Label(box, text="Your signals look close to normal - no clear cause of stress.", bg=PANEL,
                     fg=FG, font=("Helvetica", 10), anchor="w").pack(fill="x", pady=3)

        # --- short tips --------------------------------------------------------------
        tips = tk.Frame(self, bg=PANEL, padx=18, pady=10)
        tips.pack(fill="x", padx=14, pady=4)
        tk.Label(tips, text="WHAT TO DO", bg=PANEL, fg=MUTED, font=("Helvetica", 9, "bold"),
                 anchor="w").pack(fill="x")
        for tip in r.get("short_advice", []):
            tk.Label(tips, text="\u2022  " + tip, bg=PANEL, fg=FG, anchor="w", justify="left",
                     wraplength=540, font=("Helvetica", 10)).pack(fill="x", pady=2)

        # --- buttons -----------------------------------------------------------------
        btns = tk.Frame(self, bg=BG)
        btns.pack(pady=(8, 12))
        tk.Button(btns, text="Full details", bg="#2f3b3f", fg=FG, activebackground="#2f3b3f", width=14,
                  command=lambda: show_details(master, result)).pack(side="left", padx=6)
        tk.Button(btns, text="Close", bg=BLUE, fg="white", activebackground=BLUE, width=14,
                  command=self.destroy).pack(side="left", padx=6)
        info = f"Latest session: {r.get('session_id') or 'n/a'}"
        if r.get("trained_on"):
            info += f"  |  model trained on {r['trained_on']} sessions"
        elif r.get("note"):
            info += "\n" + r["note"]
        tk.Label(self, text=info, bg=BG, fg=MUTED, font=("Helvetica", 8)).pack()
        tk.Label(self, text="Automated estimate - not a medical diagnosis.", bg=BG, fg=MUTED,
                 font=("Helvetica", 8)).pack(pady=(0, 10))


def show(master, result: dict) -> ResultWindow:
    w = ResultWindow(master, result)
    w.lift()
    w.focus_force()
    return w


if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    root = tk.Tk()
    root.withdraw()
    win = show(root, data)
    win.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
