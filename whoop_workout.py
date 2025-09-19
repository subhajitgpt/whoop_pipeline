# workout_flask_genai.py  (WHOOP-style columns with header normalization)
from flask import Flask, render_template_string, request, url_for, session, redirect
import pandas as pd
import numpy as np
import os, io, base64, re
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from openai import OpenAI

# --- Config & OpenAI ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-change-me")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False

# ----------------- Helpers -----------------
def _plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64

def _bar_chart(data: dict, title=""):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(list(data.keys()), list(data.values()))
    ax.set_title(title)
    ax.set_ylabel("Value")
    plt.xticks(rotation=20)
    return _plot_to_base64(fig)

def _pie_chart(labels, sizes, title):
    import numpy as np
    fig, ax = plt.subplots(figsize=(5, 5))

    # Safety: handle empty/NaN and negative inputs
    sizes = np.array(sizes, dtype=float)
    sizes[np.isnan(sizes)] = 0
    sizes[sizes < 0] = 0
    total = sizes.sum()
    if total == 0:
        sizes = np.ones_like(sizes)  # avoid zero-division; renders equal slices
        total = sizes.sum()

    # Lightly "explode" tiny slices to make them visible
    pct = sizes / total * 100.0
    explode = np.where(pct < 5, 0.04, 0.0)  # pop out slices <5%

    # Donut look (no explicit colors)
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,                 # use legend instead of inline labels
        explode=explode,
        autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",  # show % only for >=3%
        startangle=120,
        pctdistance=0.78,
        labeldistance=1.08,
        wedgeprops=dict(width=0.42),  # ring width; donut style
        normalize=True
    )

    # Legend on the right with counts
    legend_labels = [f"{l} — {int(s)}" for l, s in zip(labels, sizes)]
    ax.legend(
        wedges,
        legend_labels,
        title=title,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False
    )

    # Center text (total)
    ax.text(0, 0, f"{int(total)}\nTotal", ha="center", va="center", fontsize=11)

    ax.set_aspect("equal")
    ax.set_title(title)
    fig.tight_layout()
    return _plot_to_base64(fig)


def fmt(n, digits=2):
    try:
        if n is None or np.isnan(n):
            return "N/A"
        return f"{float(n):,.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names and replace blanks/symbols with underscores."""
    def _norm(c):
        c = c.strip().lower()
        c = re.sub(r"[^\w]+", "_", c)        # turn spaces, () and symbols into _
        c = re.sub(r"_{2,}", "_", c).strip("_")
        return c
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    return df

# ---------- Context builder ----------
def make_context(df, agg, by_type, recent_prs):
    lines = []
    lines.append(f"Rows (workouts): {len(df)}")
    lines.append(f"Avg Duration (min): {fmt(agg['duration_min_mean'])}")
    lines.append(f"Avg Distance (km): {fmt(agg.get('distance_km_mean'))}")
    lines.append(f"Avg HR (bpm): {fmt(agg.get('avg_hr_mean'))} | Max HR (bpm): {fmt(agg.get('max_hr_mean'))}")
    lines.append(f"Avg Strain: {fmt(agg.get('strain_mean'))} | Avg Calories: {fmt(agg.get('calories_mean'))}")
    lines.append(f"Weekly Volume (min): {fmt(agg['weekly_minutes'])} | Weekly Distance (km): {fmt(agg.get('weekly_distance_km'))}")
    lines.append("Mix by activity: " + ", ".join([f"{k}={v}" for k,v in by_type.items()]))
    if recent_prs:
        lines.append("Recent PRs: " + "; ".join(recent_prs))
    # HR zone distribution if available
    if all(z in df.columns for z in ["hr_zone_1_pct","hr_zone_2_pct","hr_zone_3_pct","hr_zone_4_pct","hr_zone_5_pct"]):
        zmeans = df[["hr_zone_1_pct","hr_zone_2_pct","hr_zone_3_pct","hr_zone_4_pct","hr_zone_5_pct"]].mean(numeric_only=True)
        lines.append("Avg HR zone split (%): " + ", ".join([f"Z{i+1}={fmt(v,1)}" for i,v in enumerate(zmeans.values)]))
    return "\n".join(lines)

# ---------- Core page builder ----------
def build_page_from_csv(csv_path, prompt_text=None, answer_text=None):
    df = pd.read_csv(csv_path)
    df = normalize_headers(df)  # << normalize to underscores

    # Expected WHOOP headers after normalization
    # cycle_start_time, cycle_end_time, cycle_timezone, workout_start_time, workout_end_time,
    # duration_min, activity_name, activity_strain, energy_burned_cal, max_hr_bpm, average_hr_bpm,
    # hr_zone_1_pct ... hr_zone_5_pct, gps_enabled, distance_meters, altitude_gain_meters, altitude_change_meters

    # essential fields
    if "workout_start_time" in df.columns:
        df["_date"] = pd.to_datetime(df["workout_start_time"], errors="coerce")
    elif "cycle_start_time" in df.columns:
        df["_date"] = pd.to_datetime(df["cycle_start_time"], errors="coerce")
    else:
        raise ValueError("CSV must include 'Workout start time' or 'Cycle start time'.")

    # activity/type
    df["_type"] = df["activity_name"] if "activity_name" in df.columns else "Workout"

    # duration already in minutes
    if "duration_min" in df.columns:
        df["_duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")
    elif "duration" in df.columns:
        df["_duration_min"] = pd.to_numeric(df["duration"], errors="coerce")
    else:
        raise ValueError("CSV must include 'Duration (min)' column.")

    # optional fields mapping
    df["_avg_hr"] = pd.to_numeric(df.get("average_hr_bpm", np.nan), errors="coerce")
    df["_max_hr"] = pd.to_numeric(df.get("max_hr_bpm", np.nan), errors="coerce")
    df["_calories"] = pd.to_numeric(df.get("energy_burned_cal", np.nan), errors="coerce")
    df["_strain"] = pd.to_numeric(df.get("activity_strain", np.nan), errors="coerce")

    # meters -> km if distance present
    if "distance_meters" in df.columns:
        df["_distance_km"] = pd.to_numeric(df["distance_meters"], errors="coerce") / 1000.0
    else:
        df["_distance_km"] = np.nan

    df = df.sort_values("_date").reset_index(drop=True)
    total_rows = len(df)

    # --- Aggregations ---
    agg = {}
    
    # Safe aggregation with NaN handling
    agg["duration_min_mean"] = float(df["_duration_min"].mean()) if total_rows > 0 else 0.0
    
    distance_mean = df["_distance_km"].mean()
    agg["distance_km_mean"] = float(distance_mean) if not pd.isna(distance_mean) else None
    
    avg_hr_mean = df["_avg_hr"].mean()
    agg["avg_hr_mean"] = float(avg_hr_mean) if not pd.isna(avg_hr_mean) else None
    
    max_hr_mean = df["_max_hr"].mean()
    agg["max_hr_mean"] = float(max_hr_mean) if not pd.isna(max_hr_mean) else None
    
    calories_mean = df["_calories"].mean()
    agg["calories_mean"] = float(calories_mean) if not pd.isna(calories_mean) else None
    
    strain_mean = df["_strain"].mean()
    agg["strain_mean"] = float(strain_mean) if not pd.isna(strain_mean) else None

    # weekly totals (last 7 days of data)
    if total_rows > 0 and not df["_date"].isna().all():
        end = df["_date"].max()
        start = end - pd.Timedelta(days=6)
        mask = (df["_date"] >= start) & (df["_date"] <= end)
        agg["weekly_minutes"] = float(df.loc[mask, "_duration_min"].sum())
        agg["weekly_distance_km"] = float(df.loc[mask, "_distance_km"].sum()) if not df.loc[mask, "_distance_km"].isna().all() else 0.0
    else:
        agg["weekly_minutes"] = 0.0
        agg["weekly_distance_km"] = 0.0

    # activity mix
    by_type = df["_type"].value_counts().to_dict()

    # simple PR-like stats
    recent_prs = []
    if not df["_distance_km"].isna().all():
        max_distance = df["_distance_km"].max()
        if not pd.isna(max_distance):
            recent_prs.append(f"Best distance: {fmt(max_distance)} km")
    max_duration = df["_duration_min"].max()
    if not pd.isna(max_duration):
        recent_prs.append(f"Longest session: {fmt(max_duration)} min")

    # --- Charts ---
    averages = {
        "Dur (min)": round(agg["duration_min_mean"], 2) if agg["duration_min_mean"] else 0,
        "Dist (km)": round(agg["distance_km_mean"], 2) if agg.get("distance_km_mean") else 0,
        "Avg HR": round(agg["avg_hr_mean"], 1) if agg.get("avg_hr_mean") else 0,
        "Max HR": round(agg["max_hr_mean"], 1) if agg.get("max_hr_mean") else 0,
        "Calories": round(agg["calories_mean"], 0) if agg.get("calories_mean") else 0,
        "Strain": round(agg["strain_mean"], 1) if agg.get("strain_mean") else 0,
    }
    bar_chart = _bar_chart(averages, "Average Key Workout Metrics")

    mix_labels, mix_sizes = list(by_type.keys()), list(by_type.values())
    pie_mix = _pie_chart(mix_labels, mix_sizes, "Activity Mix") if mix_labels else None

    # best/worst by duration - safe handling
    if total_rows >= 3:
        top_long = df.nlargest(3, "_duration_min")[["_date","_type","_duration_min"]]
        low_long = df.nsmallest(3, "_duration_min")[["_date","_type","_duration_min"]]
    else:
        top_long = df.nlargest(total_rows, "_duration_min")[["_date","_type","_duration_min"]]
        low_long = df.nsmallest(total_rows, "_duration_min")[["_date","_type","_duration_min"]]
    
    best_html = top_long.rename(columns={"_date":"Date","_type":"Activity","_duration_min":"Minutes"}).to_html(index=False, escape=False)
    worst_html = low_long.rename(columns={"_date":"Date","_type":"Activity","_duration_min":"Minutes"}).to_html(index=False, escape=False)

    # Build context for ChatGPT
    context = make_context(df, agg, by_type, recent_prs)
    session["csv_path_workout"] = csv_path
    session["workout_context"] = context

    return dict(
        agg=agg,
        by_type=by_type,
        bar_chart=bar_chart,
        pie_mix=pie_mix,
        best_html=best_html,
        worst_html=worst_html,
        prompt=prompt_text or "",
        answer=answer_text or None
    )

def call_openai(context, prompt):
    if not client:
        return "[OpenAI not configured: set OPENAI_API_KEY]"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You are a coach. Be concise, numeric, actionable; balance stress and recovery."},
                {"role":"user","content": f"Workout dataset summary:\n{context}\n\nUser prompt:\n{prompt}"}
            ],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI error] {e}"

# ----------------- UI -----------------
TEMPLATE = """
<!doctype html>
<title>Workout Analyzer (WHOOP-style)</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
<style>.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre-wrap}</style>

<div class="container my-4">
  <div class="card mb-3">
    <div class="card-body">
      <h5 class="card-title">1) Upload Workout CSV
        {% if has_context %}
          <span class="badge text-bg-success ms-2">Context: True</span>
        {% else %}
          <span class="badge text-bg-secondary ms-2">Context: False</span>
        {% endif %}
      </h5>
      <form method="post" action="{{ url_for('home') }}" enctype="multipart/form-data">
        <div class="row g-2 align-items-center">
          <div class="col-auto"><input class="form-control" type="file" name="file" accept=".csv" required></div>
          <div class="col-auto"><button class="btn btn-primary" type="submit">Analyze</button></div>
          <div class="col-auto"><a class="btn btn-outline-secondary" href="{{ url_for('clear') }}">Clear context</a></div>
          <div class="col-auto"><a class="btn btn-outline-dark" href="{{ url_for('debug') }}">/debug</a></div>
        </div>
      </form>
      {% if error %}<div class="text-danger mt-2">{{ error }}</div>{% endif %}
    </div>
  </div>

  {% if agg %}
  <div class="card mb-3">
    <div class="card-body">
      <h3 class="card-title">Extracted Metrics</h3>
      <div class="row">
        <div class="col-md-7">
          <h6>Average Metrics (Bar)</h6>
          <img class="img-fluid border rounded" src="data:image/png;base64,{{ bar_chart }}" alt="Average Metrics">
        </div>
        <div class="col-md-5">
          <h6>Activity Mix</h6>
          {% if pie_mix %}
          <img class="img-fluid border rounded" src="data:image/png;base64,{{ pie_mix }}" alt="Activity Mix">
          {% else %}
          <div class="text-secondary">Activity mix not available.</div>
          {% endif %}
        </div>
      </div>

      <hr>
      <div class="row">
        <div class="col-md-6"><h6>Top 3 Longest Sessions</h6>{{ best_html|safe }}</div>
        <div class="col-md-6"><h6>Top 3 Shortest Sessions</h6>{{ worst_html|safe }}</div>
      </div>

      <hr>
      <div class="row">
        <div class="col-md-6">
          <ul class="mb-0">
            <li><b>Avg Duration:</b> {{ "%.2f"|format(agg['duration_min_mean']) }} min</li>
            <li><b>Avg Distance:</b> {{ "%.2f"|format(agg.get('distance_km_mean')) if agg.get('distance_km_mean') is not none else 'N/A' }} km</li>
            <li><b>Avg Strain:</b> {{ "%.2f"|format(agg.get('strain_mean')) if agg.get('strain_mean') is not none else 'N/A' }}</li>
          </ul>
        </div>
        <div class="col-md-6">
          <ul class="mb-0">
            <li><b>Avg HR:</b> {{ "%.1f"|format(agg.get('avg_hr_mean')) if agg.get('avg_hr_mean') is not none else 'N/A' }} bpm</li>
            <li><b>Max HR:</b> {{ "%.1f"|format(agg.get('max_hr_mean')) if agg.get('max_hr_mean') is not none else 'N/A' }} bpm</li>
            <li><b>Weekly Volume:</b> {{ "%.1f"|format(agg['weekly_minutes']) }} min | {{ "%.2f"|format(agg.get('weekly_distance_km')) if agg.get('weekly_distance_km') is not none else 'N/A' }} km</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
  {% endif %}

  <div class="card">
    <div class="card-body">
      <h5 class="card-title">2) Chat with OpenAI about this data</h5>
      {% if not has_context %}
        <div class="text-secondary">Upload a CSV first to give the assistant context. You can still type a question—I'll remind you.</div>
      {% endif %}
      <form method="post" action="{{ url_for('home') }}">
        <textarea class="form-control" name="prompt" rows="4" placeholder="E.g., Build a 7-day plan to improve stamina while keeping weekly minutes around my current average.">{{ prompt or '' }}</textarea>
        <div class="mt-3"><button class="btn btn-primary" type="submit">Ask</button></div>
      </form>
      {% if answer %}<hr><div><b>Assistant:</b></div><pre class="mono">{{ answer }}</pre>{% endif %}
    </div>
  </div>
</div>
"""

# ----------------- Routes -----------------
@app.route("/", methods=["GET", "POST"])
def home():
    # Prompt-only branch
    if request.method == "POST" and ('file' not in request.files or request.files['file'].filename == ''):
        prompt = (request.form.get("prompt") or "").strip()
        csv_path = session.get("csv_path_workout")
        context = session.get("workout_context")
        if not csv_path or not os.path.exists(csv_path):
            return render_template_string(TEMPLATE, has_context=False, error="Please upload a CSV first.")
        page_vars = build_page_from_csv(csv_path, prompt_text=prompt)
        if prompt:
            page_vars["answer"] = call_openai(context or "", prompt)
        return render_template_string(TEMPLATE, has_context=True, **page_vars)

    # Fresh upload
    if request.method == "POST" and 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return render_template_string(TEMPLATE, has_context=False, error="No selected file")
        if file and file.filename.lower().endswith('.csv'):
            filename = secure_filename(file.filename)
            csv_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(csv_path)

            try:
                page_vars = build_page_from_csv(csv_path)
                prompt = (request.form.get("prompt") or "").strip()
                if prompt:
                    page_vars["prompt"] = prompt
                    page_vars["answer"] = call_openai(session.get("workout_context",""), prompt)
                return render_template_string(TEMPLATE, has_context=True, **page_vars)
            except Exception as e:
                return render_template_string(TEMPLATE, has_context=False, error=f"Error processing CSV: {str(e)}")
        else:
            return render_template_string(TEMPLATE, has_context=False, error="Please upload a CSV file.")

    # GET
    return render_template_string(TEMPLATE, has_context=bool(session.get("workout_context")), error=None)

@app.route("/clear")
def clear():
    for k in ["csv_path_workout","workout_context"]:
        session.pop(k, None)
    return redirect(url_for("home"))

@app.route("/debug")
def debug():
    return {"has_context": bool(session.get("workout_context")),
            "csv_path": session.get("csv_path_workout")}

if __name__ == "__main__":
    print("Workout app on http://127.0.0.1:5054")
    app.run(host="127.0.0.1", port=5054, debug=True, use_reloader=False)