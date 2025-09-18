# whoop_flask_genai_2.py
from flask import Flask, render_template_string, request, url_for, session, redirect
import pandas as pd
import os, io, base64
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

# session cookie for context; OK for local dev
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-change-me")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False

# ----------------- Helpers -----------------
def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def make_bar_chart(averages):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(list(averages.keys()), list(averages.values()))  # default colors
    ax.set_ylabel('Average Value')
    ax.set_title('Average Key Metrics')
    plt.xticks(rotation=20)
    return plot_to_base64(fig)

def make_pie_chart(labels, sizes, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title(title)
    return plot_to_base64(fig)

def df_to_summary_context(df, summary_stats, recovery_dist, sleep_debt_dist, low_recovery, high_sleep_debt):
    lines = []
    lines.append(f"Rows (days): {len(df)}")
    lines.append(f"Avg Recovery: {summary_stats['Recovery_score_']['mean']}")
    lines.append(f"Avg Rest HR (bpm): {summary_stats['Resting_heart_rate_(bpm)']['mean']}")
    lines.append(f"Avg HRV (ms): {summary_stats['Heart_rate_variability_(ms)']['mean']}")
    lines.append(f"Avg Sleep Perf: {summary_stats['Sleep_performance_']['mean']}")
    if "Sleep_debt_(min)" in df.columns:
        lines.append(f"Avg Sleep Debt (min): {round(df['Sleep_debt_(min)'].mean(),2)}")
        lines.append(f"High Sleep Debt Days (>100): {len(high_sleep_debt)}")
    lines.append(f"Recovery Dist Low/Med/High: {recovery_dist}")
    if sleep_debt_dist:
        lines.append(f"Sleep Debt Dist Low/Moderate/High: {sleep_debt_dist}")
    return "\n".join(lines)

def build_page_from_csv(csv_path, prompt_text=None, answer_text=None):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    summary = {
        "Recovery_score_": df["Recovery_score_"].describe(),
        "Resting_heart_rate_(bpm)": df["Resting_heart_rate_(bpm)"].describe(),
        "Heart_rate_variability_(ms)": df["Heart_rate_variability_(ms)"].describe(),
        "Sleep_performance_": df["Sleep_performance_"].describe(),
    }
    summary_stats = {k: v.round(2).to_dict() for k, v in summary.items()}
    avg_sleep_debt = round(df["Sleep_debt_(min)"].mean(), 2) if "Sleep_debt_(min)" in df else "N/A"

    # Dists & counts
    low_recovery = df[df["Recovery_score_"] < 50][["Cycle_start_time", "Recovery_score_"]] \
        if "Cycle_start_time" in df.columns else df[df["Recovery_score_"] < 50][["Recovery_score_"]]
    high_sleep_debt = (
        df[df["Sleep_debt_(min)"] > 100][["Cycle_start_time", "Sleep_debt_(min)"]]
        if "Sleep_debt_(min)" in df and "Cycle_start_time" in df.columns
        else df[df.get("Sleep_debt_(min)", pd.Series([])) > 100][["Sleep_debt_(min)"]]
        if "Sleep_debt_(min)" in df else pd.DataFrame()
    )
    low_recovery_count = len(low_recovery)
    high_sleep_debt_count = len(high_sleep_debt)
    total_days = len(df)

    # Charts
    averages = {
        "Recovery": round(df["Recovery_score_"].mean(), 2),
        "Rest HR": round(df["Resting_heart_rate_(bpm)"].mean(), 2),
        "HRV": round(df["Heart_rate_variability_(ms)"].mean(), 2),
        "Sleep Perf": round(df["Sleep_performance_"].mean(), 2),
        "Sleep Debt": round(df["Sleep_debt_(min)"].mean(), 2) if "Sleep_debt_(min)" in df else 0,
    }
    bar_chart = make_bar_chart(averages)

    pie_low_recovery = make_pie_chart(
        ["Low Recovery (<50)", "Normal/High"],
        [low_recovery_count, total_days - low_recovery_count],
        "Low Recovery Days"
    )
    pie_high_sleep_debt = make_pie_chart(
        ["High Sleep Debt (>100)", "Normal/Low"],
        [high_sleep_debt_count, total_days - high_sleep_debt_count],
        "High Sleep Debt Days"
    )

    recovery_dist = {
        "Low": int((df["Recovery_score_"] < 50).sum()),
        "Medium": int(((df["Recovery_score_"] >= 50) & (df["Recovery_score_"] < 80)).sum()),
        "High": int((df["Recovery_score_"] >= 80).sum())
    }
    if "Sleep_debt_(min)" in df:
        sleep_debt_dist = {
            "Low": int((df["Sleep_debt_(min)"] < 30).sum()),
            "Moderate": int(((df["Sleep_debt_(min)"] >= 30) & (df["Sleep_debt_(min)"] < 100)).sum()),
            "High": int((df["Sleep_debt_(min)"] >= 100).sum())
        }
    else:
        sleep_debt_dist = {"Low": 0, "Moderate": 0, "High": 0}

    # Highlights
    best_recovery = df.nlargest(3, "Recovery_score_")[["Cycle_start_time", "Recovery_score_"]] \
        if "Cycle_start_time" in df.columns else df.nlargest(3, "Recovery_score_")[["Recovery_score_"]]
    worst_recovery = df.nsmallest(3, "Recovery_score_")[["Cycle_start_time", "Recovery_score_"]] \
        if "Cycle_start_time" in df.columns else df.nsmallest(3, "Recovery_score_")[["Recovery_score_"]]
    best_recovery_html = best_recovery.to_html(index=False) if not best_recovery.empty else "<i>None</i>"
    worst_recovery_html = worst_recovery.to_html(index=False) if not worst_recovery.empty else "<i>None</i>"

    if "Sleep_debt_(min)" in df.columns:
        highest_sleep_debt = df.nlargest(3, "Sleep_debt_(min)")[["Cycle_start_time", "Sleep_debt_(min)"]] \
            if "Cycle_start_time" in df.columns else df.nlargest(3, "Sleep_debt_(min)")[["Sleep_debt_(min)"]]
        lowest_sleep_debt = df.nsmallest(3, "Sleep_debt_(min)")[["Cycle_start_time", "Sleep_debt_(min)"]] \
            if "Cycle_start_time" in df.columns else df.nsmallest(3, "Sleep_debt_(min)")[["Sleep_debt_(min)"]]
        highest_sleep_debt_html = highest_sleep_debt.to_html(index=False) if not highest_sleep_debt.empty else "<i>None</i>"
        lowest_sleep_debt_html = lowest_sleep_debt.to_html(index=False) if not lowest_sleep_debt.empty else "<i>None</i>"
    else:
        highest_sleep_debt_html = "<i>Not available</i>"
        lowest_sleep_debt_html = "<i>Not available</i>"

    # Build + store context
    context = df_to_summary_context(df, summary_stats, recovery_dist, sleep_debt_dist, low_recovery, high_sleep_debt)
    session["csv_path"] = csv_path
    session["summary_context"] = context

    return dict(
        summary_stats=summary_stats,
        avg_sleep_debt=avg_sleep_debt,
        low_recovery_count=low_recovery_count,
        high_sleep_debt_count=high_sleep_debt_count,
        bar_chart=bar_chart,
        pie_low_recovery=pie_low_recovery,
        pie_high_sleep_debt=pie_high_sleep_debt,
        best_recovery_html=best_recovery_html,
        worst_recovery_html=worst_recovery_html,
        highest_sleep_debt_html=highest_sleep_debt_html,
        lowest_sleep_debt_html=lowest_sleep_debt_html,
        recovery_dist=recovery_dist,
        sleep_debt_dist=sleep_debt_dist,
        prompt=prompt_text or "",
        answer=answer_text or None
    )

def call_openai(context, prompt):
    if not client:
        return "[OpenAI not configured: set OPENAI_API_KEY in C:\\EUacademy\\.env]"
    try:
        # Try new Responses API first
        try:
            resp = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {"role":"system","content":"You are a health & sleep coach. Be concise, numeric, and actionable."},
                    {"role":"user","content": f"Dataset summary:\n{context}\n\nUser prompt:\n{prompt}"}
                ],
                temperature=0.2
            )
            return resp.output_text.strip()
        except Exception:
            # Fallback to chat.completions
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"You are a health & sleep coach. Be concise, numeric, and actionable."},
                    {"role":"user","content": f"Dataset summary:\n{context}\n\nUser prompt:\n{prompt}"}
                ],
                temperature=0.2
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI error] {e}"

# ----------------- Bootstrap Layout -----------------
TEMPLATE = """
<!doctype html>
<title>Health & Sleep Analyzer</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
<style>
  .monospace { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; }
</style>

<div class="container my-4">

  <div class="card mb-3">
    <div class="card-body">
      <h5 class="card-title">1) Upload WHOOP CSV
        {% if has_context %}
          <span class="badge text-bg-success ms-2">Context: True</span>
        {% else %}
          <span class="badge text-bg-secondary ms-2">Context: False</span>
        {% endif %}
      </h5>
      <form method="post" action="{{ url_for('upload_file') }}" enctype="multipart/form-data">
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

  {% if summary_stats %}
  <div class="card mb-3">
    <div class="card-body">
      <h3 class="card-title">Extracted Metrics</h3>

      <div class="row">
        <div class="col-md-7">
          <h6>Overall Metrics (Bar Chart)</h6>
          <img class="img-fluid border rounded" src="data:image/png;base64,{{ bar_chart }}" alt="Average Metrics">
        </div>
        <div class="col-md-5">
          <h6>Recovery & Sleep Debt Proportions</h6>
          <div class="row">
            <div class="col-6">
              <img class="img-fluid border rounded" src="data:image/png;base64,{{ pie_low_recovery }}" alt="Low Recovery Pie">
            </div>
            <div class="col-6">
              <img class="img-fluid border rounded" src="data:image/png;base64,{{ pie_high_sleep_debt }}" alt="High Sleep Debt Pie">
            </div>
          </div>
        </div>
      </div>

      <hr>

      <div class="row">
        <div class="col-md-6">
          <h6>Recovery Score Distribution</h6>
          <table class="table table-sm table-striped">
            <thead><tr><th>Category</th><th>Days</th></tr></thead>
            <tbody>
              <tr><td>Low (&lt;50)</td><td>{{ recovery_dist['Low'] }}</td></tr>
              <tr><td>Medium (50-79)</td><td>{{ recovery_dist['Medium'] }}</td></tr>
              <tr><td>High (&ge;80)</td><td>{{ recovery_dist['High'] }}</td></tr>
            </tbody>
          </table>
        </div>
        <div class="col-md-6">
          <h6>Sleep Debt Distribution</h6>
          <table class="table table-sm table-striped">
            <thead><tr><th>Category</th><th>Days</th></tr></thead>
            <tbody>
              <tr><td>Low (&lt;30 min)</td><td>{{ sleep_debt_dist['Low'] }}</td></tr>
              <tr><td>Moderate (30-100 min)</td><td>{{ sleep_debt_dist['Moderate'] }}</td></tr>
              <tr><td>High (&ge;100 min)</td><td>{{ sleep_debt_dist['High'] }}</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <hr>

      <h6>Highlights</h6>
      <div class="row">
        <div class="col-md-6">
          <h6>Top 3 Best Recovery Days</h6>
          {{ best_recovery_html|safe }}
          <h6 class="mt-3">Top 3 Worst Recovery Days</h6>
          {{ worst_recovery_html|safe }}
        </div>
        <div class="col-md-6">
          <h6>Top 3 Highest Sleep Debt Days</h6>
          {{ highest_sleep_debt_html|safe }}
          <h6 class="mt-3">Top 3 Lowest Sleep Debt Days</h6>
          {{ lowest_sleep_debt_html|safe }}
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
      <form method="post" action="{{ url_for('upload_file') }}">
        <textarea class="form-control" name="prompt" rows="4" placeholder="E.g., Suggest a 7-day plan to improve HRV given my averages and bad days.">{{ prompt or '' }}</textarea>
        <div class="mt-3">
          <button class="btn btn-primary" type="submit">Ask</button>
        </div>
      </form>
      {% if answer %}
        <hr>
        <div><b>Assistant:</b></div>
        <pre class="monospace">{{ answer }}</pre>
      {% endif %}
    </div>
  </div>

</div>
"""

# ----------------- Routes -----------------
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # Branch A: prompt-only (reuse stored CSV/context)
    if request.method == 'POST' and ('file' not in request.files or request.files['file'].filename == ''):
        prompt = (request.form.get("prompt") or "").strip()
        csv_path = session.get("csv_path")
        context = session.get("summary_context")
        if not csv_path or not os.path.exists(csv_path):
            return render_template_string(TEMPLATE, has_context=False, error="Please upload a CSV first.")
        # Rebuild page vars
        page_vars = build_page_from_csv(csv_path, prompt_text=prompt)
        if prompt:
            page_vars["answer"] = call_openai(context or "", prompt)
        return render_template_string(TEMPLATE, has_context=True, **page_vars)

    # Branch B: fresh upload
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return render_template_string(TEMPLATE, has_context=False, error="No selected file")
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            csv_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(csv_path)

            page_vars = build_page_from_csv(csv_path)
            # Optional immediate prompt on same request
            prompt = (request.form.get("prompt") or "").strip()
            if prompt:
                page_vars["prompt"] = prompt
                page_vars["answer"] = call_openai(session.get("summary_context",""), prompt)

            return render_template_string(TEMPLATE, has_context=True, **page_vars)
        else:
            return render_template_string(TEMPLATE, has_context=False, error="Please upload a CSV file.")

    # GET
    return render_template_string(TEMPLATE, has_context=bool(session.get("summary_context")), error=None)

@app.route("/clear")
def clear():
    for k in ["csv_path", "summary_context"]:
        session.pop(k, None)
    return redirect(url_for("upload_file"))

@app.route("/debug")
def debug():
    return {
        "has_context": bool(session.get("summary_context")),
        "csv_path": session.get("csv_path")
    }

if __name__ == '__main__':
    print("WHOOP app on http://127.0.0.1:5054")
    app.run(host="127.0.0.1", port=5054, debug=True, use_reloader=False)
