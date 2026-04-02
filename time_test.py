import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# ODIM CORE (minimal)
# ============================================================

OMEGA_MIN = 0.5
C_FLOOR = 0.05

class ODIM:
    def __init__(self, omega_s):
        self.omega_s = max(abs(omega_s), OMEGA_MIN)

    def curvature(self, x, Q2):
        return math.log(Q2 + 1.0) * (1.0 / (x + 1e-12))

    def projection_capacity(self, x, Q2, sigma):
        curv = self.curvature(x, Q2)
        return curv * sigma / (1.0 + Q2)

    def collapse_metric(self, pc):
        return pc / 1e5


# ============================================================
# CMS → ODIM mapping
# ============================================================

M_C = 1.3
SQRT_SNN = 5.36e3

def compute_x_Q2(pT, y):
    Q2 = pT**2 + M_C**2
    x = math.exp(-y) * math.sqrt(Q2) / SQRT_SNN
    return x, Q2


# ============================================================
# Load your local CSVs
# ============================================================

DATA_DIR = r"C:\Users\hillb\Documents\framwork_test\HEPData-ins2968597-v1-csv\HEPData-ins2968597-v1-csv"

CROSS_SECTION_FILES = [
    "D^0crosssectionfor2_p_T_5GeVinPbPbUPCs.csv",
    "D^0crosssectionfor5_p_T_8GeVinPbPbUPCs.csv",
    "D^0crosssectionfor8_p_T_12GeVinPbPbUPCs.csv"
]

tables = []

for fname in CROSS_SECTION_FILES:
    path = os.path.join(DATA_DIR, fname)
    print(f"Loading {fname}...")
    df = pd.read_csv(path, comment="#")
    df["source"] = fname
    tables.append(df)

data = pd.concat(tables, ignore_index=True)

# ============================================================
# Use the EXACT column names from your CSVs
# ============================================================

col_y_center = "$y$"
col_y_low = "$y$ LOW"
col_y_high = "$y$ HIGH"
col_sigma = "d$^{2}\\sigma$/dydp$_{\\mathrm{T}}$ (mb/GeV)"

# ============================================================
# pT ranges from filenames
# ============================================================

def get_pt_range(fname):
    if "2_p_T_5" in fname:
        return 2.0, 5.0
    if "5_p_T_8" in fname:
        return 5.0, 8.0
    if "8_p_T_12" in fname:
        return 8.0, 12.0
    raise ValueError("Unknown pT range in filename: " + fname)

# ============================================================
# Build analysis table
# ============================================================

odim = ODIM(omega_s=1.0)
rows = []

for _, r in data.iterrows():
    fname = r["source"]
    pt_low, pt_high = get_pt_range(fname)
    pt = 0.5 * (pt_low + pt_high)

    y_center = float(r[col_y_center])
    y_low = float(r[col_y_low])
    y_high = float(r[col_y_high])
    sigma = float(r[col_sigma])

    x, Q2 = compute_x_Q2(pt, y_center)

    curv = odim.curvature(x, Q2)
    pc = odim.projection_capacity(x, Q2, sigma)
    collapse = odim.collapse_metric(pc)

    rows.append({
        "pT_low": pt_low,
        "pT_high": pt_high,
        "pT_center": pt,
        "y_low": y_low,
        "y_high": y_high,
        "y_center": y_center,
        "x": x,
        "Q2": Q2,
        "sigma": sigma,
        "curvature": curv,
        "projection_capacity": pc,
        "collapse_metric": collapse,
        "source": fname
    })

df = pd.DataFrame(rows)
df.to_csv("cms_d0_odim_processed.csv", index=False)

print("Saved cms_d0_odim_processed.csv")

# ============================================================
# OBSERVER METRICS (Fisher-type ODIMs)
# ============================================================

def fisher_metric_O1(X):
    return np.diag([
        1.0 / (X["x"]**2 + 1e-12),
        1.0 / (X["Q2"]**2 + 1e-12),
        1.0 / (X["sigma"]**2 + 1e-12),
        10.0 / (X["curvature"]**2 + 1e-12),
        5.0 / (X["projection_capacity"]**2 + 1e-12),
        1.0 / (X["collapse_metric"]**2 + 1e-12)
    ])

def fisher_metric_O2(X):
    return np.diag([
        20.0 / (X["x"]**2 + 1e-12),
        10.0 / (X["Q2"]**2 + 1e-12),
        1.0,
        0.1,
        0.1,
        0.1
    ])

# ============================================================
# OBSERVER TIME ENGINE
# ============================================================

def compute_observer_time(df, metric_fn):
    t = [0.0]
    for i in range(len(df)-1):
        Xk = df.iloc[i]
        Xk1 = df.iloc[i+1]

        dX = np.array([
            Xk1["x"] - Xk["x"],
            Xk1["Q2"] - Xk["Q2"],
            Xk1["sigma"] - Xk["sigma"],
            Xk1["curvature"] - Xk["curvature"],
            Xk1["projection_capacity"] - Xk["projection_capacity"],
            Xk1["collapse_metric"] - Xk["collapse_metric"]
        ])

        g = metric_fn(Xk)
        ds = math.sqrt(dX.T @ g @ dX)
        t.append(t[-1] + ds)

    return t

df["t_O1"] = compute_observer_time(df, fisher_metric_O1)
df["t_O2"] = compute_observer_time(df, fisher_metric_O2)

df.to_csv("cms_d0_odim_with_time.csv", index=False)
print("Saved cms_d0_odim_with_time.csv")

# ============================================================
# PLOTS
# ============================================================

plt.figure(figsize=(7,5))
plt.plot(df["t_O1"], label="Observer O1 time")
plt.plot(df["t_O2"], label="Observer O2 time")
plt.xlabel("Step index")
plt.ylabel("Observer time")
plt.title("Observer-dependent time divergence")
plt.legend()
plt.tight_layout()
plt.savefig("observer_time_divergence.png", dpi=300)
plt.close()

print("Plots saved.")
