import os
import math
import heapq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

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
    # O1: cares about curvature + projection_capacity
    return np.diag([
        1.0 / (X["x"]**2 + 1e-12),
        1.0 / (X["Q2"]**2 + 1e-12),
        1.0 / (X["sigma"]**2 + 1e-12),
        10.0 / (X["curvature"]**2 + 1e-12),
        5.0 / (X["projection_capacity"]**2 + 1e-12),
        1.0 / (X["collapse_metric"]**2 + 1e-12)
    ])

def fisher_metric_O2(X):
    # O2: mostly x and Q2
    return np.diag([
        20.0 / (X["x"]**2 + 1e-12),
        10.0 / (X["Q2"]**2 + 1e-12),
        1.0,
        0.1,
        0.1,
        0.1
    ])

def fisher_metric_O3(X):
    # O3: cares about sigma + collapse_metric
    return np.diag([
        0.1,
        0.1,
        20.0 / (X["sigma"]**2 + 1e-12),
        1.0,
        1.0,
        15.0 / (X["collapse_metric"]**2 + 1e-12)
    ])

# ============================================================
# OBSERVER TIME ENGINE
# ============================================================

def compute_observer_time(df, metric_fn):
    t = [0.0]
    X = df[["x","Q2","sigma","curvature","projection_capacity","collapse_metric"]].values
    for i in range(len(df)-1):
        Xk = X[i]
        Xk1 = X[i+1]
        dX = Xk1 - Xk
        g = metric_fn(df.iloc[i])
        ds = math.sqrt(dX.T @ g @ dX)
        t.append(t[-1] + ds)
    return t

df["t_O1"] = compute_observer_time(df, fisher_metric_O1)
df["t_O2"] = compute_observer_time(df, fisher_metric_O2)
df["t_O3"] = compute_observer_time(df, fisher_metric_O3)

df.to_csv("cms_d0_odim_with_time.csv", index=False)
print("Saved cms_d0_odim_with_time.csv")

# ============================================================
# CLOCK SUBMANIFOLD (1D clock metric)
# ============================================================

def clock_metric(X):
    # Clock lives mostly in projection_capacity space
    return np.diag([
        0.0,  # x
        0.0,  # Q2
        0.0,  # sigma
        0.0,  # curvature
        50.0 / (X["projection_capacity"]**2 + 1e-12),
        0.0   # collapse_metric
    ])

def compute_clock_time(df, metric_fn):
    theta = [0.0]
    X = df[["x","Q2","sigma","curvature","projection_capacity","collapse_metric"]].values
    for i in range(len(df)-1):
        Xk = X[i]
        Xk1 = X[i+1]
        dX = Xk1 - Xk
        g = metric_fn(df.iloc[i])
        ds = math.sqrt(dX.T @ g @ dX)
        theta.append(theta[-1] + ds)
    return theta

df["theta_clock"] = compute_clock_time(df, clock_metric)

# ============================================================
# NORMALIZE OBSERVER TIMES
# ============================================================

def normalize_series(s):
    s_min = s.min()
    s_max = s.max()
    if s_max - s_min < 1e-12:
        return s * 0.0
    return (s - s_min) / (s_max - s_min)

df["t_O1_norm"] = normalize_series(df["t_O1"])
df["t_O2_norm"] = normalize_series(df["t_O2"])
df["t_O3_norm"] = normalize_series(df["t_O3"])

# ============================================================
# LOCAL CURVATURE UNDER EACH METRIC
# ============================================================

def local_curvature(df, metric_fn):
    kappas = [0.0]
    X = df[["x","Q2","sigma","curvature","projection_capacity","collapse_metric"]].values
    for k in range(1, len(df)-1):
        X_prev = X[k-1]
        X_curr = X[k]
        X_next = X[k+1]

        v1 = X_curr - X_prev
        v2 = X_next - X_curr

        g = metric_fn(df.iloc[k])

        v1_g = g @ v1
        v2_g = g @ v2
        norm_v1 = math.sqrt(v1 @ v1_g + 1e-18)
        norm_v2 = math.sqrt(v2 @ v2_g + 1e-18)

        cos_theta = (v1 @ v2_g) / (norm_v1 * norm_v2 + 1e-18)
        cos_theta = max(min(cos_theta, 1.0), -1.0)
        theta = math.acos(cos_theta)
        kappas.append(theta)

    kappas.append(0.0)
    return kappas

df["kappa_O1"] = local_curvature(df, fisher_metric_O1)
df["kappa_O2"] = local_curvature(df, fisher_metric_O2)
df["kappa_O3"] = local_curvature(df, fisher_metric_O3)

# ============================================================
# MDS 2D EMBEDDING (O2 METRIC)
# ============================================================

def pairwise_distances_metric(df, metric_fn):
    X = df[["x","Q2","sigma","curvature","projection_capacity","collapse_metric"]].values
    n = len(X)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dX = X[j] - X[i]
            g = metric_fn(df.iloc[i])
            ds = math.sqrt(dX.T @ g @ dX)
            D[i,j] = D[j,i] = ds
    return D

D_O2 = pairwise_distances_metric(df, fisher_metric_O2)

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
X_2d = mds.fit_transform(D_O2)

df["mds_x"] = X_2d[:,0]
df["mds_y"] = X_2d[:,1]

# ============================================================
# GEODESIC SOLVER (O2 METRIC, DISCRETE)
# ============================================================

def build_graph(df, metric_fn, k_neighbors=2):
    X = df[["x","Q2","sigma","curvature","projection_capacity","collapse_metric"]].values
    n = len(X)
    graph = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(max(0, i-k_neighbors), min(n, i+k_neighbors+1)):
            if i == j:
                continue
            dX = X[j] - X[i]
            g = metric_fn(df.iloc[i])
            w = math.sqrt(dX.T @ g @ dX)
            graph[i].append((j, w))
    return graph

def dijkstra(graph, start, end):
    dist = {node: float("inf") for node in graph}
    prev = {node: None for node in graph}
    dist[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == end:
            break
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    path = []
    u = end
    while u is not None:
        path.append(u)
        u = prev[u]
    path.reverse()
    return path, dist[end]

graph_O2 = build_graph(df, fisher_metric_O2, k_neighbors=2)
geo_path, geo_length = dijkstra(graph_O2, start=0, end=len(df)-1)
print("Geodesic path indices (O2 metric):", geo_path)
print("Geodesic length (O2 metric):", geo_length)

# ============================================================
# PLOTS
# ============================================================

# Observer time divergence
plt.figure(figsize=(7,5))
plt.plot(df["t_O1"], label="Observer O1 time")
plt.plot(df["t_O2"], label="Observer O2 time")
plt.plot(df["t_O3"], label="Observer O3 time")
plt.xlabel("Step index")
plt.ylabel("Observer time")
plt.title("Observer-dependent time divergence (O1, O2, O3)")
plt.legend()
plt.tight_layout()
plt.savefig("observer_time_divergence_O1_O2_O3.png", dpi=300)
plt.close()

# Clock vs observers
plt.figure(figsize=(7,5))
plt.plot(df["theta_clock"], label="Clock θ", color="black", linestyle="--")
plt.plot(df["t_O1"], label="t_O1", alpha=0.7)
plt.plot(df["t_O2"], label="t_O2", alpha=0.7)
plt.plot(df["t_O3"], label="t_O3", alpha=0.7)
plt.xlabel("Step index")
plt.ylabel("Time parameter")
plt.title("Clock submanifold vs observer times")
plt.legend()
plt.tight_layout()
plt.savefig("clock_vs_observers.png", dpi=300)
plt.close()

# Normalized observer times
plt.figure(figsize=(7,5))
plt.plot(df["t_O1_norm"], label="O1 (norm)")
plt.plot(df["t_O2_norm"], label="O2 (norm)")
plt.plot(df["t_O3_norm"], label="O3 (norm)")
plt.xlabel("Step index")
plt.ylabel("Normalized time")
plt.title("Normalized observer times (rates of aging)")
plt.legend()
plt.tight_layout()
plt.savefig("observer_time_normalized.png", dpi=300)
plt.close()

# Local curvature
plt.figure(figsize=(7,5))
plt.plot(df["kappa_O1"], label="κ_O1")
plt.plot(df["kappa_O2"], label="κ_O2")
plt.plot(df["kappa_O3"], label="κ_O3")
plt.xlabel("Step index")
plt.ylabel("Local curvature (angle)")
plt.title("Local trajectory curvature under each observer metric")
plt.legend()
plt.tight_layout()
plt.savefig("local_curvature_observers.png", dpi=300)
plt.close()

# MDS embedding (O2 metric)
plt.figure(figsize=(6,5))
sc = plt.scatter(df["mds_x"], df["mds_y"], c=df["t_O2_norm"], cmap="viridis")
for i, row in df.iterrows():
    plt.text(row["mds_x"], row["mds_y"], str(i))
plt.xlabel("MDS-1 (O2 metric)")
plt.ylabel("MDS-2 (O2 metric)")
plt.title("2D embedding of trajectory (O2 ODIM)")
plt.colorbar(sc, label="t_O2_norm")
plt.tight_layout()
plt.savefig("mds_O2_trajectory.png", dpi=300)
plt.close()

print("All plots saved.")
