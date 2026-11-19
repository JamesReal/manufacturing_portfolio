"""Production Planning & Constraint Optimization — synthetic brewery scheduling decision-support prototype.
Inputs: batches.csv, tanks.csv (same folder). Times are hours after Nov 19 2026 06:00."""
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd, plotly.express as px, streamlit as st

st.set_page_config(page_title="Brewery Production Planner", layout="wide", page_icon="🏭")
T0 = datetime(2026, 11, 19, 6)
dt = lambda h: T0 + timedelta(hours=float(h))
fmt = lambda h: f"{dt(h):%b %d %H:%M}"
WEIGHT = {1: 1, 2: 2, 3: 4, 4: 10}
PRI = {1: "Low", 2: "Medium", 3: "High", 4: "Urgent"}
# family: (duration_h, needs_glycol, cleaning_h)
FAM = {"West Coast IPA": (120, False, 8), "Hazy IPA": (132, True, 12), "Lager": (240, True, 8), "Pilsner": (216, True, 8),
       "Amber Ale": (144, False, 6), "Stout/Porter": (168, False, 10), "Sour/Wild": (300, False, 24)}
# scoring constants (used by the engine AND quoted in the UI so the explanation can never drift from the code)
PT_IDLE, PT_CHANGE, PT_WASTE, PT_LOAD, OVERFLOW_DAYS = 0.05, 8, 0.1, 0.02, 7
STRAT = {"Priority first": "highest priority first, ties broken by earliest due date",
         "Earliest due date": "earliest due date first, ties broken by higher priority",
         "Minimize changeovers": "grouped by product family (alphabetical), then by release date",
         "Balanced utilization": "by release date, and tanks that are already heavily booked get a load penalty"}

BATCH_CSV = Path(__file__).parent / "batches.csv"
REQUIRED = ["batch", "product", "family", "volume", "priority", "release", "due", "duration", "cleaning"]

def load_batches():
    """Read batches.csv (hours are offsets from Nov 19 2026 06:00; priority 1=Low 2=Medium 3=High)."""
    if not BATCH_CSV.exists():
        st.error(f"Missing {BATCH_CSV.name} in the app folder."); st.stop()
    df = pd.read_csv(BATCH_CSV)
    missing = [c for c in REQUIRED if c not in df.columns]
    badfam = sorted(set(df.family) - set(FAM)) if not missing else []
    if missing or badfam or df.batch.duplicated().any():
        st.error(f"batches.csv problem — missing columns: {missing or 'none'}; unknown families: {badfam or 'none'}; "
                 f"duplicate IDs: {sorted(df.batch[df.batch.duplicated()]) or 'none'}"); st.stop()
    return df[REQUIRED]

TANK_CSV = Path(__file__).parent / "tanks.csv"

def load_tanks():
    """Read tanks.csv. families: ';'-separated list. maintenance: ';'-separated 'start-end' hour offsets (blank = none).
    occupied: same format; time ranges a tank is already holding an existing batch (blocked for new work)."""
    if not TANK_CSV.exists():
        st.error(f"Missing {TANK_CSV.name} in the app folder."); st.stop()
    df = pd.read_csv(TANK_CSV, keep_default_na=False)
    need = ["tank", "capacity", "families", "glycol", "maintenance"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        st.error(f"tanks.csv missing columns: {missing}"); st.stop()
    tanks, problems = [], []
    for r in df.itertuples():
        fams = [f.strip() for f in str(r.families).split(";") if f.strip()]
        bad = [f for f in fams if f not in FAM]
        def parse(txt, label):
            try:
                return [tuple(float(x) for x in w.split("-")) for w in str(txt).split(";") if w.strip()]
            except ValueError:
                problems.append(f"{r.tank}: unreadable {label} '{txt}'"); return []
        wins = parse(r.maintenance, "maintenance")
        occ = parse(getattr(r, "occupied", ""), "occupied")
        if bad: problems.append(f"{r.tank}: unknown families {bad}")
        tanks.append(dict(tank=r.tank, capacity=int(r.capacity), families=fams,
                          glycol=str(r.glycol).strip().lower() in ("true", "1", "yes", "y"), windows=wins, occupied=occ))
    if df.tank.duplicated().any(): problems.append("duplicate tank IDs")
    if problems:
        st.error("tanks.csv problem — " + "; ".join(problems)); st.stop()
    return tanks

@st.cache_data
def make_data():
    return load_tanks(), load_batches()

# ---------------------------------------------------------------- engine
def schedule(tanks, b, strategy, H):
    order = {"Priority first": lambda r: (-r.priority, r.due), "Earliest due date": lambda r: (r.due, -r.priority),
             "Minimize changeovers": lambda r: (r.family, r.release), "Balanced utilization": lambda r: (r.release, r.due)}[strategy]
    state = {t["tank"]: dict(free=0, last=None, load=0) for t in tanks}
    out, bad = [], []
    for seq, r in enumerate(sorted(b.itertuples(), key=order), 1):
        cands = [t for t in tanks if r.family in t["families"] and t["capacity"] >= r.volume and (t["glycol"] or not FAM[r.family][1])]
        if not cands:
            fam_ok = [t for t in tanks if r.family in t["families"]]
            mx = max([t["capacity"] for t in fam_ok], default=0)
            why = (f"Volume {r.volume} bbl exceeds the largest compatible tank ({mx} bbl)" if r.volume > mx
                   else "No compatible tank has the required glycol/temperature control" if fam_ok else "No tank accepts this product family")
            bad.append(dict(batch=r.batch, product=r.product, volume=r.volume, priority=PRI[r.priority], reason=why)); continue
        scored = []
        for t in cands:
            s = state[t["tank"]]; ready = max(r.release, s["free"]); start = ready
            moved = True
            while moved:
                moved = False
                for ws, we in t["windows"] + t["occupied"]:
                    if start < we and start + r.duration + r.cleaning > ws:
                        start, moved = we, True
            end = start + r.duration
            parts = dict(late=max(0, end - r.due) * WEIGHT[r.priority],
                         idle=((start - ready) + (start - s["free"])) * PT_IDLE,
                         change=PT_CHANGE if s["last"] not in (None, r.family) else 0,
                         waste=(t["capacity"] - r.volume) * PT_WASTE,
                         load=s["load"] * PT_LOAD if strategy == "Balanced utilization" else 0)
            scored.append((sum(parts.values()), t["tank"], start, end, parts))
        scored.sort(key=lambda x: x[0])
        sc, tk, start, end, parts = scored[0]
        run_ = scored[1] if len(scored) > 1 else None
        if end + r.cleaning > H + OVERFLOW_DAYS * 24:
            bad.append(dict(batch=r.batch, product=r.product, volume=r.volume, priority=PRI[r.priority],
                            reason="Compatible tanks are fully booked within the planning horizon")); continue
        state[tk].update(free=end + r.cleaning, last=r.family, load=state[tk]["load"] + r.duration + r.cleaning)
        slack = r.due - end
        out.append(dict(seq=seq, batch=r.batch, product=r.product, family=r.family, tank=tk, volume=r.volume, priority=PRI[r.priority],
                        pw=WEIGHT[r.priority], start=start, end=end, clean_end=end + r.cleaning, cleaning=r.cleaning, release=r.release,
                        due=r.due, late_h=max(0, -slack), status="Late" if slack < 0 else "At risk" if slack < 24 else "On time",
                        n_compat=len(cands), score=round(sc, 1), **{f"pts_{k}": round(v, 1) for k, v in parts.items()},
                        runner_up=run_[1] if run_ else "—", runner_score=round(run_[0], 1) if run_ else None))
    return pd.DataFrame(out), pd.DataFrame(bad)

def metrics(s, bad, tanks, n, H):
    c = lambda x: min(max(x, 0), H)
    rows = []
    for t in tanks:
        x = s[s.tank == t["tank"]]
        brew = sum(c(e) - c(a) for a, e in zip(x.start, x.end)); cl = sum(c(e) - c(a) for a, e in zip(x.end, x.clean_end))
        occ = sum(c(e) - c(a) for a, e in t["occupied"])
        rows.append(dict(tank=t["tank"], brewing_h=brew, cleaning_h=cl, occupied_h=occ, batches=len(x), util=(brew + cl + occ) / H))
    td = pd.DataFrame(rows).set_index("tank")
    return dict(orders=n, scheduled=len(s), unscheduled=len(bad), late=int((s.status == "Late").sum()), at_risk=int((s.status == "At risk").sum()),
                util=round(100 * td.util.mean(), 1), clean_h=int(s.cleaning.sum()), tank_df=td, tank_util=td.util)

@st.cache_data
def run(strategy, H):
    tanks, b = make_data(); s, bad = schedule(tanks, b, strategy, H)
    return dict(s=s, bad=bad, m=metrics(s, bad, tanks, len(b), H), tanks=tanks, b=b)

# ---------------------------------------------------------------- explanations of every number
D = dict(orders="Count of rows in batches.csv.",
         scheduled="Batches that received a tank and start time that satisfy all hard constraints.",
         unscheduled=f"Batches with no compatible tank (family, size or glycol), or whose compatible tanks are booked more than {OVERFLOW_DAYS} days past the horizon.",
         late="Scheduled batches whose projected finish (start + duration) is after their due date.",
         at_risk="Scheduled batches that finish less than 24 h before their due date.",
         util="Per tank: (brewing h + cleaning h + already-occupied h, counted inside the horizon) ÷ horizon hours. The KPI is the simple mean over all tanks. Maintenance/outage time is not counted as use.")
CFG = dict(Batch="Batch ID from batches.csv", Tank="Tank the scheduler assigned (see 'How tanks were prioritized' on page 03)",
           Priority="Priority from batches.csv: 1 Low, 2 Medium, 3 High, 4 Urgent. Weight used in scoring: " + ", ".join(f"{PRI[k]}={v}" for k, v in WEIGHT.items()),
           Due="Due date = batches.csv due (hours after Nov 19 06:00) shown as a date",
           Finish="Projected finish = scheduled start + fermentation duration (cleaning not included)",
           LateH="Late by (h) = projected finish − due date, in hours",
           LateD="Late by (days) = late hours ÷ 24")

# ---------------------------------------------------------------- insights (rule-based, from the results)
def insights(R, H):
    s, bad, m, out = R["s"], R["bad"], R["m"], []
    td = m["tank_df"]; late = s[s.status == "Late"]
    if len(late):
        cons = late[late.n_compat <= 5]; top = td.util.sort_values(ascending=False)
        how = "Rule: if ≥ half of the late batches could use ≤ 5 tanks (size/glycol/family limits), the constraint is tank access; otherwise total capacity."
        if len(cons) >= len(late) / 2:
            out.append(("Tank availability is the primary constraint",
                        f"{len(cons)} of {len(late)} late batches can use only ≤5 tanks, while average utilization is {m['util']}%. The constraint is compatible-tank access, not total capacity.",
                        f"Shift flexible lower-priority batches off {top.index[0]} ({top.iloc[0]*100:.0f}% loaded) and keep it free for late-prone high-priority work.", how))
        else:
            out.append(("Total capacity is the constraint", f"{len(late)} batches are late with average utilization at {m['util']}%.",
                        "Extend the planning horizon, add tank time, or defer low-priority orders.", how))
    hi, lo = s[s.pw >= 4], s[s.pw == 1]
    if len(late) and len(lo) and lo.late_h.mean() > 72:
        out.append(("Priority vs on-time tradeoff", f"Low-priority batches absorb the delay: they average {lo.late_h.mean():.0f} h late"
                    + (f" vs {hi.late_h.mean():.0f} h for high priority." if len(hi) else "."),
                    "Confirm customers on low-priority orders can tolerate this, or renegotiate dates.",
                    "Mean of (finish − due, floored at 0) for Low-priority batches (weight 1) vs High/Urgent (weight ≥ 4)."))
    for _, r in bad.iterrows():
        out.append((f"{r.batch} cannot be scheduled", r.reason + ".", "Resolve via an equipment change (larger tank), splitting the batch, or renegotiating the order.",
                    "Hard-constraint check: tank must accept the family, hold the volume and have glycol if required."))
    idle = td[td.util < .5]
    if len(idle):
        out.append(("Under-utilised equipment", f"{', '.join(idle.index)} run below 50% utilization over {H // 24} days.",
                    "Review whether compatibility rules on these tanks can be broadened to absorb overflow.",
                    "Utilization = (brewing + cleaning + occupied hours) ÷ horizon hours, flagged when < 50%."))
    brew = (s.end - s.start).sum()
    out.append(("Cleaning overhead", f"{m['clean_h']} tank-hours go to cleaning ({m['clean_h'] / max(1, brew) * 100:.0f}% of brewing hours).",
                "Look at shorter or sequenced cleaning to recover capacity.", "Σ cleaning hours of scheduled batches ÷ Σ brewing hours of scheduled batches."))
    return out

# ---------------------------------------------------------------- appearance
THEMES = {"Soft Indigo": ("radial-gradient(1100px 600px at 8% -10%,#e4e0ff 0%,#f7f8fc 60%)", "#5b3df5"),
          "Sea Breeze": ("radial-gradient(1100px 600px at 90% -10%,#cdeffa 0%,#f6fbfd 62%)", "#0891b2"),
          "Amber Brewery": ("radial-gradient(1100px 600px at 10% -10%,#ffe8b8 0%,#fffaf0 62%)", "#d97706")}
def style(theme, accent, glass, glow, grad, compact):
    bg = THEMES[theme][0]
    st.markdown(f"""<style>
    .stApp{{background:{bg};background-attachment:fixed}}
    [data-testid=stSidebar]{{background:rgba(255,255,255,.75);backdrop-filter:blur(14px);border-right:1px solid {accent}44}}
    {'h1{background:linear-gradient(90deg,' + accent + ',#ff5ca8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800}' if grad else ''}
    [data-testid=stMetric]{{border:1px solid {accent}55;border-radius:14px;padding:12px 16px;
      {'background:rgba(255,255,255,.75);backdrop-filter:blur(8px);' if glass else ''}{'box-shadow:0 4px 18px ' + accent + '33;' if glow else ''}}}
    [data-testid=stMetricValue]{{color:{accent}}}
    [data-testid=stVerticalBlockBorderWrapper]{{border-radius:14px;border-color:{accent}66}}
    .stButton>button{{border:1px solid {accent};border-radius:10px;{'box-shadow:0 2px 10px ' + accent + '55;' if glow else ''}}}
    .block-container{{padding-top:{'1rem' if compact else '3rem'}}}
    </style>""", unsafe_allow_html=True)

def gantt(R, H, mode, v0, span, accent):
    rows = []
    for r in R["s"].itertuples():
        rows.append(dict(tank=r.tank, Start=dt(r.start), Finish=dt(r.end), Kind=r.family if mode == "family" else r.status, Label=r.batch,
                         Detail=f"{r.batch} — {r.product}<br>Vol {r.volume} bbl · Priority {r.priority}<br>{fmt(r.start)} → {fmt(r.end)} · Due {fmt(r.due)}<br>Status {r.status}"))
        rows.append(dict(tank=r.tank, Start=dt(r.end), Finish=dt(r.clean_end), Kind="Cleaning", Label="", Detail=f"Cleaning after {r.batch}: {r.cleaning} h"))
    for t in R["tanks"]:
        for ws, we in t["windows"]: rows.append(dict(tank=t["tank"], Start=dt(ws), Finish=dt(we), Kind="Maintenance/outage", Label="", Detail=f"{t['tank']} unavailable"))
        for ws, we in t["occupied"]: rows.append(dict(tank=t["tank"], Start=dt(ws), Finish=dt(we), Kind="Occupied (existing)", Label="", Detail=f"{t['tank']} holds an existing batch"))
    cmap = {"Cleaning": "#546e7a", "Maintenance/outage": "#37474f", "Occupied (existing)": "#8d6e63",
            "On time": "#2ecc71", "At risk": "#f9a825", "Late": "#ff4d6d"}
    f = px.timeline(pd.DataFrame(rows), x_start="Start", x_end="Finish", y="tank", color="Kind", text="Label", hover_name="Detail",
                    color_discrete_map=cmap, color_discrete_sequence=px.colors.qualitative.Vivid, category_orders={"tank": [t["tank"] for t in R["tanks"]]})
    f.update_yaxes(autorange="reversed", title=None); f.update_traces(textposition="inside", hovertemplate="%{hovertext}<extra></extra>")
    f.update_xaxes(range=[dt(v0 * 24), dt((v0 + span) * 24)])
    f.add_vline(x=dt(H), line_dash="dot", line_color=accent, annotation_text="Horizon end")
    f.update_layout(template="plotly_white", height=580, legend_title=None, margin=dict(t=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.6)")
    return f

# ---------------------------------------------------------------- sidebar
page = st.sidebar.radio("Navigate", ["01 Operations Overview", "02 Production Orders", "03 Schedule & Equipment", "04 Insights & Recommendations"])
H_days = st.sidebar.select_slider("Planning horizon (days)", [14, 21, 28, 42, 60, 90], 21,
                                  help="How far ahead the scheduler plans. Utilization percentages are measured against this many hours.")
H = H_days * 24
strategy = st.sidebar.selectbox("Scheduling strategy", list(STRAT), help="Controls the order in which batches are placed. See page 03 for details.")
if st.sidebar.button("🔄 Reload batches & tanks CSVs"):
    st.cache_data.clear(); st.rerun()
with st.sidebar.expander("🎨 Appearance", expanded=False):
    theme = st.selectbox("Theme", list(THEMES))
    accent = st.color_picker("Accent colour", THEMES[theme][1], key=f"acc_{theme}")
    glass, glow = st.toggle("Glass cards", True), st.toggle("Neon glow", True)
    grad, compact = st.toggle("Gradient title", True), st.toggle("Compact layout", False)
style(theme, accent, glass, glow, grad, compact)
st.sidebar.caption("All data is synthetic. Decision-support prototype — not a production MES.")

R = run(strategy, H); s, bad, m, td = R["s"], R["bad"], R["m"], R["m"]["tank_df"]
st.title("Production Planning & Constraint Optimization")
st.caption(f"Planning horizon: {H_days} days from {T0:%b %d, %Y} · strategy: {strategy}")

# ---------------------------------------------------------------- pages
if page.startswith("01"):
    cols = st.columns(6)
    for col, (k, key, v) in zip(cols, [("Production orders", "orders", m["orders"]), ("Scheduled", "scheduled", m["scheduled"]), ("Unscheduled", "unscheduled", m["unscheduled"]),
                                       ("Late", "late", m["late"]), ("At risk", "at_risk", m["at_risk"]), ("Avg tank utilization", "util", f"{m['util']}%")]):
        col.metric(k, v, help=D[key])
    a, b_ = st.columns([3, 2])
    u = (td[["brewing_h", "cleaning_h", "occupied_h"]] / H * 100).round(1).rename(columns={"brewing_h": "Brewing", "cleaning_h": "Cleaning", "occupied_h": "Already occupied"})
    f = px.bar(u.reset_index().melt("tank", var_name="Use", value_name="% of horizon"), x="tank", y="% of horizon", color="Use", title="Tank utilization", range_y=[0, 100],
               color_discrete_map={"Brewing": accent, "Cleaning": "#546e7a", "Already occupied": "#8d6e63"})
    f.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.6)", xaxis_title=None)
    a.plotly_chart(f, width="stretch")
    a.caption(f"Each bar segment = hours of that kind inside the {H} h horizon ÷ {H} h × 100. Bar total = tank utilization. Maintenance/outages are excluded.")
    with b_:
        top = td.util.sort_values(ascending=False); name = top.index[0]; x = td.loc[name]
        late_on = s[(s.tank == name) & (s.status == "Late")]; nxt = f"; next most loaded is {top.index[1]} at {top.iloc[1]*100:.0f}%" if len(top) > 1 else ""
        st.subheader("Critical bottleneck")
        st.metric(name, f"{top.iloc[0]*100:.0f}%", help="Bottleneck = the tank with the highest utilization (brewing + cleaning + occupied hours ÷ horizon hours).")
        st.write(f"**Why:** {name} is the most heavily loaded tank. {x.brewing_h:.0f} h brewing + {x.cleaning_h:.0f} h cleaning + {x.occupied_h:.0f} h already occupied "
                 f"= {x.brewing_h + x.cleaning_h + x.occupied_h:.0f} of {H} h ({top.iloc[0]*100:.0f}%), across {int(x.batches)} batches. "
                 f"{len(late_on)} of them finish late ({late_on.late_h.sum():.0f} h total){nxt}. Anything that needs this tank queues behind that load.")
    st.subheader("Late batches")
    late = s[s.status == "Late"].sort_values("late_h", ascending=False)
    if late.empty: st.success("No late batches in this plan.")
    else:
        t1, t2 = st.columns([2, 3])
        by = late.groupby("tank").agg(batches=("batch", "count"), total=("late_h", "sum"), worst=("late_h", "max")).sort_values("total", ascending=False).reset_index()
        t1.dataframe(by.rename(columns={"tank": "Tank", "batches": "Late batches", "total": "Total late (h)", "worst": "Worst late (h)"}).round(0), hide_index=True, width="stretch",
                     column_config={"Late batches": st.column_config.Column(help="Count of this tank's batches with finish > due"),
                                    "Total late (h)": st.column_config.Column(help="Sum of (finish − due) over this tank's late batches"),
                                    "Worst late (h)": st.column_config.Column(help="Largest (finish − due) among this tank's late batches")})
        lt = pd.DataFrame({"Batch": late.batch, "Tank": late.tank, "Priority": late.priority, "Due": late.due.map(fmt), "Finish": late.end.map(fmt),
                           "LateH": late.late_h.round(0).astype(int), "LateD": (late.late_h / 24).round(1)})
        t2.dataframe(lt, hide_index=True, width="stretch", column_config={k: st.column_config.Column(v if k not in ("LateH", "LateD") else v, help=v)
                     for k, v in CFG.items() if k in lt.columns} | {"LateH": st.column_config.NumberColumn("Late by (h)", help=CFG["LateH"]),
                                                                    "LateD": st.column_config.NumberColumn("Late by (days)", help=CFG["LateD"])})
    if len(bad):
        st.subheader("Unscheduled batches"); st.dataframe(bad, hide_index=True, width="stretch",
            column_config={"reason": st.column_config.Column("Reason", help="Which hard constraint failed (family / size / glycol / horizon capacity)")})

elif page.startswith("02"):
    f1, f2 = st.columns(2)
    pri = f1.multiselect("Priority", list(PRI.values()), default=list(PRI.values()), help="Show only batches whose batches.csv priority is selected.")
    fam = f2.multiselect("Product family", list(FAM), default=list(FAM), help="Show only batches of the selected product families.")
    t = R["b"].assign(Priority=R["b"].priority.map(PRI), Due=R["b"].due.map(fmt), Release=R["b"].release.map(fmt))
    t = t[t.Priority.isin(pri) & t.family.isin(fam)].merge(s[["batch", "tank", "status"]], how="left", on="batch")
    t["status"] = t.status.fillna("UNSCHEDULED"); t = t.reset_index(drop=True)
    view = t[["batch", "product", "volume", "Priority", "Release", "Due", "duration", "tank", "status"]]
    st.caption("Click a row to open its batch detail below. Hover any column header for how it is derived.")
    ev = st.dataframe(view, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", column_config={
        "batch": st.column_config.Column("Batch", help=CFG["Batch"]), "product": st.column_config.Column("Product", help="Product name from batches.csv"),
        "volume": st.column_config.Column("Volume (bbl)", help="Barrels per batch, from batches.csv"), "Priority": st.column_config.Column(help=CFG["Priority"]),
        "Release": st.column_config.Column(help="Earliest allowed start = batches.csv release hours after Nov 19 06:00"), "Due": st.column_config.Column(help=CFG["Due"]),
        "duration": st.column_config.Column("Duration (h)", help="Hours the tank is occupied brewing this batch, from batches.csv"),
        "tank": st.column_config.Column("Tank", help=CFG["Tank"]),
        "status": st.column_config.Column("Status", help="Late: finish > due · At risk: finish < 24 h before due · On time: otherwise · UNSCHEDULED: no feasible tank")})
    rows = ev.selection.rows
    with st.container(border=True):
        if not rows: st.info("No batch selected — click a row in the table above.")
        else:
            sel = view.iloc[rows[0]].batch; r = R["b"].set_index("batch").loc[sel]; sc = s[s.batch == sel]
            d1, d2 = st.columns(2)
            with d1:
                st.markdown(f"### {sel} — {r['product']}")
                st.write(f"**Volume:** {r.volume} bbl  \n**Priority:** {PRI[r.priority]} (weight {WEIGHT[r.priority]})  \n**Fermentation duration:** {r.duration} h  \n**Cleaning after:** {r.cleaning} h  \n"
                         f"**Earliest start:** {fmt(r.release)}  \n**Due:** {fmt(r.due)}  \n**Needs temperature control:** {'Yes' if FAM[r.family][1] else 'No'}")
            with d2:
                if len(sc):
                    x = sc.iloc[0]
                    (st.success if x.status == "On time" else st.warning)(f"Assigned **{x.tank}** · {fmt(x.start)} → {fmt(x.end)} · {x.status}" + (f" ({x.late_h:.0f} h late)" if x.late_h else ""))
                    st.write(f"**Why {x.tank}:** score {x.score} (lower wins) vs runner-up {x.runner_up} at {x.runner_score}. Compatible tanks: "
                             + ", ".join(tt["tank"] for tt in R["tanks"] if r.family in tt["families"] and tt["capacity"] >= r.volume and (tt["glycol"] or not FAM[r.family][1])))
                else: st.error("Unscheduled: " + bad.set_index("batch").loc[sel, "reason"])

elif page.startswith("03"):
    c1, c2, c3 = st.columns([1, 1, 2])
    mode = c1.radio("Colour by", ["family", "status"], horizontal=True, help="Family: product family of each batch. Status: on time / at risk / late versus its due date.")
    spans = [d for d in (7, 14, 21, 28, 42, 60, 90) if d <= H_days] or [H_days]
    span = c2.select_slider("View window (days)", spans, spans[-1], help="How many days of the schedule are visible at once.")
    v0 = c3.slider("Window starts on day", 0, max(0, H_days - span), 0, help="Day offset from Nov 19; slide to look further into the future.") if H_days > span else 0
    st.plotly_chart(gantt(R, H, mode, v0, span, accent), width="stretch")
    st.caption("Reading the chart: coloured bars = brewing from scheduled start to finish · slate = cleaning after the batch (cleaning hours from batches.csv) · "
               "dark grey = maintenance/outage · brown = tank already occupied · dotted line = end of the planning horizon. Hover a bar for its dates.")
    tk = pd.DataFrame({"Tank": td.index, "Capacity (bbl)": [t["capacity"] for t in R["tanks"]], "Glycol": [t["glycol"] for t in R["tanks"]], "Batches": td.batches.astype(int),
                       "Brewing h": td.brewing_h.round(0), "Cleaning h": td.cleaning_h.round(0), "Occupied h": td.occupied_h.round(0), "Utilization %": (td.util * 100).round(0)})
    st.dataframe(tk, hide_index=True, width="stretch", column_config={
        "Capacity (bbl)": st.column_config.Column(help="Maximum batch size, tanks.csv"), "Glycol": st.column_config.Column(help="Temperature control available, tanks.csv"),
        "Batches": st.column_config.Column(help="Number of batches assigned to this tank"), "Brewing h": st.column_config.Column(help=f"Σ batch durations inside the {H} h horizon"),
        "Cleaning h": st.column_config.Column(help="Σ cleaning hours after each batch inside the horizon"), "Occupied h": st.column_config.Column(help="Σ of tanks.csv 'occupied' ranges inside the horizon"),
        "Utilization %": st.column_config.Column(help=D["util"])})
    st.header("How tanks were prioritized")
    st.markdown(f"""
**Step 1 – Batch order.** Strategy **{strategy}**: batches are placed one at a time, {STRAT[strategy]}. The *Seq* column below shows that order.

**Step 2 – Hard filter.** A tank is only a candidate if it accepts the batch's product family, its capacity ≥ batch volume, and it has glycol if the product needs it.
Batches with no candidate are unscheduled.

**Step 3 – Earliest start per candidate.** Start = later of the batch's release time and the moment the tank is free after its previous batch's cleaning.
If the batch plus its cleaning would overlap a maintenance/outage window or an existing occupied range, the start moves to the end of that range.

**Step 4 – Score every candidate; the lowest score wins.**
| Component | Points |
|---|---|
| Lateness | hours past due date × priority weight ({', '.join(f'{PRI[k]} {v}' for k, v in WEIGHT.items())}) |
| Idle / waiting | (hours the tank sits empty before the start + hours pushed back by blocked windows) × {PT_IDLE} |
| Changeover | +{PT_CHANGE} if the tank's previous batch was a different product family |
| Wasted capacity | (tank capacity − batch volume) in bbl × {PT_WASTE}, so small batches avoid big tanks |
| Load (Balanced strategy only) | hours already booked on the tank × {PT_LOAD} |

**Step 5 – Commit.** The winning tank is booked through brewing + cleaning, and the next batch sees that tank as busy.
""")
    log = s.sort_values("seq")[["seq", "batch", "priority", "tank", "score", "pts_late", "pts_idle", "pts_change", "pts_waste", "pts_load", "runner_up", "runner_score", "n_compat"]]
    st.dataframe(log, hide_index=True, width="stretch", column_config={
        "seq": st.column_config.Column("Seq", help="Order in which the strategy placed this batch"), "batch": "Batch", "priority": "Priority", "tank": st.column_config.Column("Chosen tank", help="Lowest total score among candidates"),
        "score": st.column_config.Column("Score", help="Sum of the five components to the right; lower is better"),
        "pts_late": st.column_config.Column("Lateness pts", help="Hours late × priority weight"), "pts_idle": st.column_config.Column("Idle pts", help=f"(idle gap + blocked-window wait) h × {PT_IDLE}"),
        "pts_change": st.column_config.Column("Changeover pts", help=f"{PT_CHANGE} if the family differs from the tank's previous batch"),
        "pts_waste": st.column_config.Column("Waste pts", help=f"(capacity − volume) × {PT_WASTE}"), "pts_load": st.column_config.Column("Load pts", help=f"Booked hours × {PT_LOAD}, Balanced strategy only"),
        "runner_up": st.column_config.Column("Runner-up", help="Second-lowest scoring candidate tank"), "runner_score": st.column_config.Column("Runner-up score", help="Total score of the runner-up"),
        "n_compat": st.column_config.Column("Candidates", help="Number of tanks that passed the hard filter")})

else:
    st.caption("Generated from the engine's results, not hardcoded. Each card says how it was calculated.")
    for h, why, act, how in insights(R, H):
        with st.container(border=True):
            st.markdown(f"#### {h}"); st.write(why); st.markdown(f"**Recommended action:** {act}"); st.caption(f"How calculated — {how}")
