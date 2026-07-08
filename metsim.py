"""metsim.py - Patient-level what-if simulation for metabolic syndrome risk.
Loads simulator_artifacts.pkl (trained sex-stratified models, calibrators,
plausibility constraints) and lets a user explore how plausibility-constrained
changes in modifiable factors affect an individual's calibrated risk, with a
paired SHAP explanation. No patient-level data is required at runtime.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle
try:
    import shap
except ImportError:
    shap = None


class MetSimulator:
    def __init__(self, artifacts_path="simulator_artifacts.pkl"):
        with open(artifacts_path, "rb") as f:
            self.A = pickle.load(f)

    def _key(self, sex, has_lab):
        s = "Female" if str(sex).lower() in ("0", "female", "f") else "Male"
        cfg = "B" if has_lab else "A"
        return f"{s}_{cfg}", (0 if s == "Female" else 1), cfg

    def _vector(self, key, profile):
        preds = self.A["preds"][key]
        defaults = self.A["defaults"][key]
        x = np.array([float(profile.get(f, defaults[f])) for f in preds], dtype=float)
        return x, preds

    def simulate_patient(self, profile, sex, has_lab=True, modifications=None):
        key, sexo, cfg = self._key(sex, has_lab)
        cal = self.A["calibrators"][key]
        x0, preds = self._vector(key, profile)
        risk0 = float(cal.predict_proba(x0.reshape(1, -1))[0, 1])

        x1, applied = x0.copy(), {}
        for v, new_val in (modifications or {}).items():
            if v not in self.A["simulable"][cfg]:
                raise ValueError(f"{v} not simulable in Config {cfg}. "
                                 f"Allowed: {self.A['simulable'][cfg]}")
            i = preds.index(v)
            delta = float(new_val) - x0[i]
            lo, hi = self.A["plausible_delta"][sexo].get(v, (-np.inf, np.inf))
            clipped = float(np.clip(delta, lo, hi))
            if abs(clipped - delta) > 1e-9:
                print(f"[warn] {v}: delta {delta:+.2f} outside plausible "
                      f"[{lo:+.2f}, {hi:+.2f}]; clipped to {clipped:+.2f}")
            x1[i] = x0[i] + clipped
            applied[v] = {"from": x0[i], "to": x1[i], "delta": clipped}
        risk1 = float(cal.predict_proba(x1.reshape(1, -1))[0, 1])

        sv0 = sv1 = None
        if shap is not None:
            expl = shap.TreeExplainer(self.A["models"][key])
            def _sv(x):
                s = expl.shap_values(x.reshape(1, -1))
                if isinstance(s, list):
                    s = s[1]
                return np.asarray(s).reshape(-1)
            sv0, sv1 = _sv(x0), _sv(x1)

        return {"config": cfg, "sex": "female" if sexo == 0 else "male",
                "baseline_risk": risk0, "modified_risk": risk1,
                "risk_change": risk1 - risk0, "applied": applied,
                "features": preds, "shap_baseline": sv0, "shap_modified": sv1,
                "simulable": self.A["simulable"][cfg]}

    def plot_simulation(self, res, top=8, save=None):
        import matplotlib.pyplot as plt
        if res["shap_baseline"] is None:
            raise RuntimeError("Install shap to use plot_simulation.")
        feats = res["features"]
        sv0, sv1 = res["shap_baseline"], res["shap_modified"]
        order = np.argsort(np.abs(sv0))[::-1][:top][::-1]
        names = [feats[i] for i in order]
        fig, (axS, axR) = plt.subplots(1, 2, figsize=(13, 5),
                                       gridspec_kw={"width_ratios": [2.3, 1]})
        y = np.arange(len(names)); h = 0.38
        axS.barh(y + h/2, sv0[order], height=h, color="#9aa7b8", label="Baseline")
        axS.barh(y - h/2, sv1[order], height=h, color="crimson", label="After modification")
        axS.set_yticks(y); axS.set_yticklabels(names); axS.axvline(0, color="gray", lw=0.8)
        axS.set_xlabel("SHAP value (impact on model output)")
        axS.set_title("Paired SHAP comparison"); axS.legend(fontsize=9, loc="lower right")
        axR.bar(["Baseline", "Modified"],
                [res["baseline_risk"]*100, res["modified_risk"]*100],
                color=["#9aa7b8", "crimson"], width=0.6)
        for i, val in enumerate([res["baseline_risk"]*100, res["modified_risk"]*100]):
            axR.text(i, val + 1.5, f"{val:.0f}%", ha="center", fontsize=11)
        axR.set_ylim(0, 100); axR.set_ylabel("Calibrated MetS risk (%)")
        mods = ", ".join([f"{v} {a['delta']:+.1f}" for v, a in res["applied"].items()])
        axR.set_title("Predicted risk\n(" + mods + ")", fontsize=10)
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150, bbox_inches="tight")
        plt.show()
