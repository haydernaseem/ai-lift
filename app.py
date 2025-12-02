import io
import datetime
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

app = Flask(__name__)
CORS(app)


# ========= helpers =========

def read_table_from_file(file_storage):
    filename = file_storage.filename.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(file_storage)
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(file_storage)
    else:
        raise ValueError("Only CSV / Excel files are supported.")
    return df


def find_col(df, candidates):
    cols = [c.lower().strip() for c in df.columns]
    for cand_group in candidates:
        for c in cand_group:
            if c.lower() in cols:
                return df.columns[cols.index(c.lower())]
    return None


def safe_list(x):
    return [] if x is None else list(map(float, x))


def build_base_summary(df):
    summary = []
    summary.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        summary.append(f"Numeric columns: {', '.join(numeric_cols[:6])}"
                       + ("..." if len(numeric_cols) > 6 else ""))
    else:
        summary.append("No numeric columns detected.")
    return summary


# ========= core analysis for each lift type =========

def analyze_gas_lift(df):
    # Try to detect required columns
    t_col = find_col(df, [["time", "date", "day"]])
    q_col = find_col(df, [["q_oil", "oil_rate", "rate", "qo"]])
    inj_col = find_col(df, [["q_gas_inj", "gas_injection", "gas_rate", "qginj"]])
    whp_col = find_col(df, [["whp", "wellhead_pressure"]])
    bhp_col = find_col(df, [["bhp", "bottomhole_pressure"]])

    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    # simple time index
    if t_col is None:
        numeric["t"] = np.arange(len(numeric))
        t_col = "t"
    else:
        numeric["t"] = np.arange(len(numeric))

    # default columns if not found
    if q_col is None and numeric.shape[1] > 0:
        q_col = numeric.columns[0]
    if inj_col is None and numeric.shape[1] > 1:
        inj_col = numeric.columns[1]

    t = numeric["t"].values
    q = numeric[q_col].values if q_col in numeric.columns else np.zeros_like(t)
    inj = numeric[inj_col].values if inj_col in numeric.columns else np.zeros_like(t)

    # crude optimization: find best average rate within 5 quantiles of injection
    if inj.std() > 0:
        qtiles = np.linspace(0.2, 0.9, 6)
        best_q = None
        best_mean_rate = -1
        for qt in qtiles:
            thr = np.quantile(inj, qt)
            mask = inj <= thr
            if mask.sum() < 5:
                continue
            mean_rate = q[mask].mean()
            if mean_rate > best_mean_rate:
                best_mean_rate = mean_rate
                best_q = thr
        opt_inj = float(best_q) if best_q is not None else float(np.nan)
    else:
        opt_inj = float(np.nan)

    # time series plot
    time_series = {
        "t": safe_list(t),
        "q_oil": safe_list(q),
        "inj_gas": safe_list(inj)
    }

    # optimization curve (inj vs q)
    inj_sorted_idx = np.argsort(inj)
    inj_sorted = inj[inj_sorted_idx]
    q_sorted = q[inj_sorted_idx]

    opt_curve = {
        "x": safe_list(inj_sorted),
        "y": safe_list(q_sorted),
        "xlabel": inj_col or "Gas Injection Rate",
        "ylabel": q_col or "Oil Rate",
        "title": "Gas Injection vs Oil Rate"
    }

    summary = build_base_summary(df)
    summary.append("Lift Type: Gas Lift")
    if not np.isnan(opt_inj):
        summary.append(f"Suggested optimal gas injection ~ {opt_inj:.2f} (units as in file).")

    recs = []
    recs.append("Reduce gas injection above the AI suggested optimum to avoid over-injection.")
    if whp_col or bhp_col:
        recs.append("Check WHP/BHP trends for unstable gradients or rapid fluctuations.")
    recs.append("Use the valve map and gradient curves (next version) to refine the optimal injection window.")

    risks = []
    if inj.std() == 0:
        risks.append("Gas injection is almost constant – no systematic optimization attempts detected.")
    if q.std() < 1e-3:
        risks.append("Oil rate shows minimal variation – possible metering issues or unstable well behavior.")

    metrics = {
        "avg_oil_rate": float(np.nanmean(q)) if len(q) else None,
        "max_oil_rate": float(np.nanmax(q)) if len(q) else None,
        "avg_gas_injection": float(np.nanmean(inj)) if len(inj) else None,
        "opt_gas_injection": opt_inj
    }

    return {
        "lift_type": "Gas Lift",
        "summary": summary,
        "recommendations": recs,
        "risks": risks,
        "plots": {
            "time_series": time_series,
            "opt_curve": opt_curve
        },
        "metrics": metrics
    }


def analyze_esp(df):
    t_col = find_col(df, [["time", "date", "day"]])
    q_col = find_col(df, [["q_oil", "oil_rate", "rate", "qo"]])
    freq_col = find_col(df, [["freq_hz", "frequency", "hz", "vfd"]])
    intake_col = find_col(df, [["intake_pressure", "pi", "pump_intake"]])
    discharge_col = find_col(df, [["discharge_pressure", "pd", "pump_discharge"]])

    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    if t_col is None:
        numeric["t"] = np.arange(len(numeric))
        t_col = "t"
    else:
        numeric["t"] = np.arange(len(numeric))

    if q_col is None and numeric.shape[1] > 0:
        q_col = numeric.columns[0]
    if freq_col is None and numeric.shape[1] > 1:
        freq_col = numeric.columns[1]

    t = numeric["t"].values
    q = numeric[q_col].values if q_col in numeric.columns else np.zeros_like(t)
    f = numeric[freq_col].values if freq_col in numeric.columns else np.zeros_like(t)

    # Fit simple quadratic: q = a f^2 + b f + c
    if len(f) >= 5 and f.std() > 0:
        try:
            coeffs = np.polyfit(f, q, 2)
            a, b, c = coeffs
            if a != 0:
                f_opt = -b / (2 * a)
            else:
                f_opt = np.nan
        except Exception:
            f_opt = np.nan
    else:
        f_opt = np.nan

    time_series = {
        "t": safe_list(t),
        "q_oil": safe_list(q),
        "freq": safe_list(f)
    }

    f_sorted_idx = np.argsort(f)
    f_sorted = f[f_sorted_idx]
    q_sorted = q[f_sorted_idx]
    opt_curve = {
        "x": safe_list(f_sorted),
        "y": safe_list(q_sorted),
        "xlabel": freq_col or "Frequency (Hz)",
        "ylabel": q_col or "Oil Rate",
        "title": "Frequency vs Oil Rate"
    }

    summary = build_base_summary(df)
    summary.append("Lift Type: ESP")
    if not np.isnan(f_opt):
        summary.append(f"AI suggested optimal operating frequency ≈ {f_opt:.2f} Hz (within data range).")

    recs = []
    recs.append("Operate close to the AI suggested frequency window while monitoring motor load and vibration.")
    if intake_col or discharge_col:
        recs.append("Track pump intake/discharge pressures to avoid gas lock and overload conditions.")
    recs.append("Use the time series chart to detect sudden drops in rate (possible failures or gas interference).")

    risks = []
    if f.std() == 0:
        risks.append("Frequency is almost constant – no VFD optimization patterns detected.")
    if q.std() < 1e-3:
        risks.append("Production is nearly flat; check measurement accuracy or well stability.")
    if not np.isnan(f_opt) and (f_opt < np.min(f) or f_opt > np.max(f)):
        risks.append("AI optimal frequency lies outside the historical operating range – validate before applying.")

    metrics = {
        "avg_oil_rate": float(np.nanmean(q)) if len(q) else None,
        "max_oil_rate": float(np.nanmax(q)) if len(q) else None,
        "avg_frequency": float(np.nanmean(f)) if len(f) else None,
        "opt_frequency": float(f_opt) if not np.isnan(f_opt) else None
    }

    return {
        "lift_type": "ESP",
        "summary": summary,
        "recommendations": recs,
        "risks": risks,
        "plots": {
            "time_series": time_series,
            "opt_curve": opt_curve
        },
        "metrics": metrics
    }


def analyze_pcp(df):
    t_col = find_col(df, [["time", "date", "day"]])
    q_col = find_col(df, [["q_oil", "oil_rate", "rate", "qo"]])
    rpm_col = find_col(df, [["rpm", "speed", "spinner"]])
    torque_col = find_col(df, [["torque", "tq"]])

    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    if t_col is None:
        numeric["t"] = np.arange(len(numeric))
        t_col = "t"
    else:
        numeric["t"] = np.arange(len(numeric))

    if q_col is None and numeric.shape[1] > 0:
        q_col = numeric.columns[0]
    if rpm_col is None and numeric.shape[1] > 1:
        rpm_col = numeric.columns[1]

    t = numeric["t"].values
    q = numeric[q_col].values if q_col in numeric.columns else np.zeros_like(t)
    rpm = numeric[rpm_col].values if rpm_col in numeric.columns else np.zeros_like(t)

    # simple linear relation q ~ a * rpm + b
    if len(rpm) >= 3 and rpm.std() > 0:
        try:
            coeffs = np.polyfit(rpm, q, 1)
            a, b = coeffs
            rpm_min, rpm_max = np.min(rpm), np.max(rpm)
            # propose point where marginal gain starts diminishing: 80% of max RPM
            rpm_opt = rpm_min + 0.8 * (rpm_max - rpm_min)
        except Exception:
            rpm_opt = np.nan
    else:
        rpm_opt = np.nan

    time_series = {
        "t": safe_list(t),
        "q_oil": safe_list(q),
        "rpm": safe_list(rpm)
    }

    rpm_sorted_idx = np.argsort(rpm)
    rpm_sorted = rpm[rpm_sorted_idx]
    q_sorted = q[rpm_sorted_idx]
    opt_curve = {
        "x": safe_list(rpm_sorted),
        "y": safe_list(q_sorted),
        "xlabel": rpm_col or "RPM",
        "ylabel": q_col or "Oil Rate",
        "title": "RPM vs Oil Rate"
    }

    summary = build_base_summary(df)
    summary.append("Lift Type: PCP")
    if not np.isnan(rpm_opt):
        summary.append(f"AI suggested RPM window center ≈ {rpm_opt:.2f} RPM (80% of max historical speed).")

    recs = []
    recs.append("Avoid running consistently at max RPM to reduce wear on the elastomer and rod string.")
    if torque_col:
        recs.append("Monitor torque to catch early signs of sanding or plugging events.")
    recs.append("Use the RPM vs rate curve to identify a sweet spot between rate increase and mechanical risk.")

    risks = []
    if rpm.std() == 0:
        risks.append("PCP RPM is almost constant – no dynamic optimization detected.")
    if q.std() < 1e-3:
        risks.append("Production is nearly flat; check pump condition and inflow performance.")
    metrics = {
        "avg_oil_rate": float(np.nanmean(q)) if len(q) else None,
        "max_oil_rate": float(np.nanmax(q)) if len(q) else None,
        "avg_rpm": float(np.nanmean(rpm)) if len(rpm) else None,
        "opt_rpm": float(rpm_opt) if not np.isnan(rpm_opt) else None
    }

    return {
        "lift_type": "PCP",
        "summary": summary,
        "recommendations": recs,
        "risks": risks,
        "plots": {
            "time_series": time_series,
            "opt_curve": opt_curve
        },
        "metrics": metrics
    }


def auto_detect_lift_type(df):
    cols = [c.lower() for c in df.columns]

    if any(k in " ".join(cols) for k in ["gas_inj", "q_gas", "glv", "annulus"]):
        return "gas_lift"
    if any(k in " ".join(cols) for k in ["freq", "vfd", "esp", "intake_pressure", "discharge_pressure"]):
        return "esp"
    if any(k in " ".join(cols) for k in ["rpm", "pcp", "torque", "rod"]):
        return "pcp"
    # fallback default
    return "esp"


# ========= API endpoints =========

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze
    form-data:
      - file: CSV or Excel
      - lift_type: "auto" / "gas_lift" / "esp" / "pcp"
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded."}), 400

        file = request.files["file"]
        lift_type = request.form.get("lift_type", "auto").lower().strip()

        df = read_table_from_file(file)

        if lift_type == "auto":
            lift_type = auto_detect_lift_type(df)

        if lift_type in ["gas_lift", "gas lift"]:
            result = analyze_gas_lift(df)
        elif lift_type == "esp":
            result = analyze_esp(df)
        elif lift_type == "pcp":
            result = analyze_pcp(df)
        else:
            return jsonify({"error": f"Unknown lift type: {lift_type}"}), 400

        result["lift_type_code"] = lift_type
        result["generated_at"] = datetime.datetime.utcnow().isoformat() + "Z"

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download-report", methods=["POST"])
def download_report():
    """
    POST /download-report
    JSON body:
      { "analysis": { ...result from /analyze... } }
    Returns: PDF file.
    """
    try:
        data = request.get_json()
        if not data or "analysis" not in data:
            return jsonify({"error": "Missing analysis payload."}), 400

        analysis = data["analysis"]

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        title = "OILNOVA – AI Lift Optimization Report"
        story.append(Paragraph(title, styles["Title"]))
        story.append(Spacer(1, 12))

        lift_type = analysis.get("lift_type", "Unknown")
        story.append(Paragraph(f"Lift Type: <b>{lift_type}</b>", styles["Normal"]))
        story.append(Spacer(1, 6))

        gen_time = analysis.get("generated_at", "")
        if gen_time:
            story.append(Paragraph(f"Generated at (UTC): {gen_time}", styles["Normal"]))
            story.append(Spacer(1, 12))

        # Summary
        story.append(Paragraph("<b>Summary</b>", styles["Heading2"]))
        for line in analysis.get("summary", []):
            story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 12))

        # Metrics
        metrics = analysis.get("metrics", {})
        if metrics:
            story.append(Paragraph("<b>Key Metrics</b>", styles["Heading2"]))
            table_data = [["Metric", "Value"]]
            for k, v in metrics.items():
                if v is None:
                    val_str = "-"
                else:
                    val_str = f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
                table_data.append([k, val_str])
            t = Table(table_data, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#020617")),
            ]))
            story.append(t)
            story.append(Spacer(1, 12))

        # Recommendations
        recs = analysis.get("recommendations", [])
        if recs:
            story.append(Paragraph("<b>AI Recommendations</b>", styles["Heading2"]))
            for r in recs:
                story.append(Paragraph(f"• {r}", styles["Normal"]))
            story.append(Spacer(1, 12))

        # Risks
        risks = analysis.get("risks", [])
        if risks:
            story.append(Paragraph("<b>Risks & Alerts</b>", styles["Heading2"]))
            for r in risks:
                story.append(Paragraph(f"• {r}", styles["Normal"]))
            story.append(Spacer(1, 12))

        story.append(Paragraph(
            "This report was generated automatically by OILNOVA AI – Lift Optimization Engine.",
            styles["Italic"]
        ))

        doc.build(story)
        buffer.seek(0)

        filename = f"OILNOVA_AI_Lift_Report_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype="application/pdf"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # For local testing
    app.run(host="0.0.0.0", port=5000, debug=True)
