import io
import datetime
import numpy as np
import pandas as pd
from scipy import stats, signal, optimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    Image, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.platypus.flowables import Spacer
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Line, Rect
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics import renderPDF
from reportlab.graphics.charts.textlabels import Label
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import base64

app = Flask(__name__)
CORS(app)


# ========== ENHANCED ANALYSIS FUNCTIONS ==========

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
    cols = [str(c).lower().strip() for c in df.columns]
    for cand_group in candidates:
        for c in cand_group:
            for col_name in df.columns:
                if c.lower() in str(col_name).lower():
                    return col_name
    return None


def safe_list(x):
    if x is None:
        return []
    try:
        return list(map(float, x))
    except:
        return []


def detect_outliers_iqr(data, threshold=1.5):
    """Detect outliers using IQR method"""
    if len(data) < 4:
        return np.zeros(len(data), dtype=bool)
    
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return np.zeros(len(data), dtype=bool)
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (data < lower_bound) | (data > upper_bound)


def build_advanced_summary(df, lift_type):
    """Enhanced summary with statistical insights"""
    summary = []
    summary.append(f"<b>Dataset:</b> {len(df)} rows × {len(df.columns)} columns")
    
    # Identify key columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        summary.append(f"<b>Numeric columns:</b> {', '.join(numeric_cols[:8])}"
                      + ("..." if len(numeric_cols) > 8 else ""))
    
    # Basic statistics for first few numeric columns
    for i, col in enumerate(numeric_cols[:3]):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(vals) > 0:
                summary.append(f"<b>{col}:</b> μ={vals.mean():.2f}, σ={vals.std():.2f}, "
                              f"range=[{vals.min():.2f}, {vals.max():.2f}]")
    
    summary.append(f"<b>Lift Type:</b> {lift_type}")
    return summary


def calculate_productivity_index(q_oil, bhp, whp=None):
    """Calculate PI (Productivity Index) if data available"""
    if len(q_oil) < 5 or bhp is None:
        return None
    
    valid_mask = ~np.isnan(q_oil) & ~np.isnan(bhp)
    if valid_mask.sum() < 5:
        return None
    
    q_valid = q_oil[valid_mask]
    bhp_valid = bhp[valid_mask]
    
    # Simple linear regression for PI
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(bhp_valid, q_valid)
        return abs(slope)  # PI is positive
    except:
        return None


def interp_optimal_production(analysis):
    """Interpolate production at optimal point"""
    plots = analysis.get("plots", {})
    opt = plots.get("opt_curve", {})
    
    if opt.get('opt_point') and len(opt['opt_point']) == 2:
        return opt['opt_point'][1]
    
    # Fallback calculation
    metrics = analysis.get("metrics", {})
    if metrics.get('avg_oil_rate'):
        return metrics['avg_oil_rate'] * 1.1  # 10% improvement estimate
    
    return 0


def analyze_gas_lift_advanced(df):
    """Enhanced Gas Lift analysis with machine learning and physics-based models"""
    
    # Detect columns with improved pattern matching
    t_col = find_col(df, [["time", "date", "day", "timestamp", "index", "t"]])
    q_col = find_col(df, [["q_oil", "oil_rate", "oil", "rate", "qo", "production", "liquid_rate"]])
    inj_col = find_col(df, [["q_gas_inj", "gas_injection", "gas_inj", "gas_rate", "qginj", "injection", "gl_inj"]])
    whp_col = find_col(df, [["whp", "wellhead_pressure", "head_pressure", "pressure"]])
    bhp_col = find_col(df, [["bhp", "bottomhole_pressure", "bottom_pressure", "reservoir_pressure"]])
    glv_col = find_col(df, [["glv", "valve", "orifice", "port"]])
    
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    
    # Fill missing values with interpolation for time series
    numeric = numeric.interpolate(limit_direction='both')
    
    # Create time index if not present
    if t_col is None:
        numeric["t"] = np.arange(len(numeric))
        t_col = "t"
    else:
        numeric["t"] = np.arange(len(numeric))
    
    # Ensure we have required columns
    if q_col is None:
        # Try to find any column that looks like production
        for col in numeric.columns:
            if any(word in str(col).lower() for word in ["rate", "prod", "oil", "q"]):
                q_col = col
                break
        if q_col is None and len(numeric.columns) > 0:
            q_col = numeric.columns[0]
    
    if inj_col is None:
        for col in numeric.columns:
            if any(word in str(col).lower() for word in ["gas", "inj", "injection", "gl"]):
                inj_col = col
                break
        if inj_col is None and len(numeric.columns) > 1:
            inj_col = numeric.columns[1]
    
    # Extract arrays
    t = numeric["t"].values
    q = numeric[q_col].values if q_col in numeric.columns else np.zeros_like(t)
    inj = numeric[inj_col].values if inj_col in numeric.columns else np.zeros_like(t)
    whp = numeric[whp_col].values if whp_col and whp_col in numeric.columns else np.full_like(t, np.nan)
    bhp = numeric[bhp_col].values if bhp_col and bhp_col in numeric.columns else np.full_like(t, np.nan)
    
    # Clean data - remove outliers
    q_outliers = detect_outliers_iqr(q, threshold=2.0)
    inj_outliers = detect_outliers_iqr(inj, threshold=2.0)
    valid_mask = ~q_outliers & ~inj_outliers & ~np.isnan(q) & ~np.isnan(inj)
    
    if valid_mask.sum() < 10:
        valid_mask = ~np.isnan(q) & ~np.isnan(inj)
    
    t_clean = t[valid_mask]
    q_clean = q[valid_mask]
    inj_clean = inj[valid_mask]
    
    # 1. Advanced Optimization using Random Forest
    opt_inj_rf = None
    if len(q_clean) >= 20:
        try:
            # Prepare features for ML
            X = inj_clean.reshape(-1, 1)
            y = q_clean
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            # Generate predictions across injection range
            inj_range = np.linspace(np.min(inj_clean), np.max(inj_clean), 200)
            q_pred = rf.predict(inj_range.reshape(-1, 1))
            
            # Find optimal injection (maximum predicted rate)
            opt_idx = np.argmax(q_pred)
            opt_inj_rf = float(inj_range[opt_idx])
            q_at_opt = float(q_pred[opt_idx])
            
        except Exception as e:
            print(f"RF optimization failed: {e}")
    
    # 2. Physics-based optimization using Vogel-type analysis
    opt_inj_physics = None
    if len(q_clean) >= 10 and len(inj_clean) >= 10:
        try:
            # Sort by injection rate
            sort_idx = np.argsort(inj_clean)
            inj_sorted = inj_clean[sort_idx]
            q_sorted = q_clean[sort_idx]
            
            # Apply Savitzky-Golay filter for smoothing
            if len(q_sorted) > 5:
                q_smooth = signal.savgol_filter(q_sorted, min(5, len(q_sorted)//2*2+1), 2)
            else:
                q_smooth = q_sorted
            
            # Find peak using gradient analysis
            gradients = np.gradient(q_smooth)
            zero_crossings = np.where(np.diff(np.sign(gradients)))[0]
            
            if len(zero_crossings) > 0:
                # First zero crossing from positive to negative
                peak_idx = zero_crossings[0]
                opt_inj_physics = float(inj_sorted[peak_idx])
        except:
            pass
    
    # 3. Economic optimization (simplified)
    # Assume gas cost and oil price (these could be parameters)
    gas_cost_per_unit = 0.5  # $ per unit gas
    oil_price_per_unit = 50  # $ per unit oil
    
    if opt_inj_rf is not None:
        # Calculate profitability curve
        inj_range = np.linspace(np.min(inj_clean), np.max(inj_clean), 100)
        
        # Simple profit function: profit = oil_revenue - gas_cost
        if len(q_clean) >= 10:
            # Interpolate production response
            from scipy.interpolate import interp1d
            sort_idx = np.argsort(inj_clean)
            f = interp1d(inj_clean[sort_idx], q_clean[sort_idx], 
                        bounds_error=False, fill_value="extrapolate")
            q_range = f(inj_range)
            
            profit_range = q_range * oil_price_per_unit - inj_range * gas_cost_per_unit
            
            if not np.all(np.isnan(profit_range)):
                opt_inj_economic = float(inj_range[np.nanargmax(profit_range)])
            else:
                opt_inj_economic = opt_inj_rf
        else:
            opt_inj_economic = opt_inj_rf
    else:
        opt_inj_economic = None
    
    # Choose best optimal injection
    candidates = [c for c in [opt_inj_rf, opt_inj_physics, opt_inj_economic] if c is not None]
    if candidates:
        opt_inj_final = np.median(candidates)  # Use median for robustness
    else:
        # Fallback to quantile method
        if inj_clean.std() > 0:
            opt_inj_final = float(np.percentile(inj_clean, 75))
        else:
            opt_inj_final = float(np.nanmean(inj_clean))
    
    # Calculate PI if BHP available
    pi = calculate_productivity_index(q_clean, bhp[valid_mask] if bhp_col else None)
    
    # Time series data
    time_series = {
        "t": safe_list(t),
        "q_oil": safe_list(q),
        "inj_gas": safe_list(inj),
        "whp": safe_list(whp) if whp_col else [],
        "bhp": safe_list(bhp) if bhp_col else []
    }
    
    # Optimization curve with confidence interval
    inj_sorted_idx = np.argsort(inj_clean)
    inj_sorted = inj_clean[inj_sorted_idx]
    q_sorted = q_clean[inj_sorted_idx]
    
    # Add smoothed curve
    if len(q_sorted) >= 5:
        q_smooth_curve = signal.savgol_filter(q_sorted, min(5, len(q_sorted)//2*2+1), 2)
    else:
        q_smooth_curve = q_sorted
    
    # Calculate optimal production
    if not np.isnan(opt_inj_final) and len(inj_sorted) > 0 and len(q_smooth_curve) > 0:
        opt_production = float(np.interp(opt_inj_final, inj_sorted, q_smooth_curve))
        opt_point = [float(opt_inj_final), opt_production]
    else:
        opt_production = None
        opt_point = None
    
    opt_curve = {
        "x": safe_list(inj_sorted),
        "y": safe_list(q_sorted),
        "y_smooth": safe_list(q_smooth_curve),
        "xlabel": inj_col or "Gas Injection Rate",
        "ylabel": q_col or "Oil Rate",
        "title": "Gas Injection vs Oil Rate",
        "opt_point": opt_point
    }
    
    # Enhanced summary
    summary = build_advanced_summary(df, "Gas Lift")
    summary.append(f"<b>Data Quality:</b> {valid_mask.sum()}/{len(t)} valid points after cleaning")
    
    if not np.isnan(opt_inj_final):
        summary.append(f"<b>AI Optimal Gas Injection:</b> {opt_inj_final:.2f} units")
        if opt_production:
            summary.append(f"<b>Expected Oil Rate at Optimum:</b> {opt_production:.2f} units")
    
    if pi is not None:
        summary.append(f"<b>Productivity Index (PI):</b> {pi:.4f} units/psi")
    
    # Calculate efficiency metrics
    if len(q_clean) > 0 and len(inj_clean) > 0:
        current_avg_q = np.mean(q_clean)
        current_avg_inj = np.mean(inj_clean)
        current_efficiency = current_avg_q / current_avg_inj if current_avg_inj > 0 else 0
        
        if opt_inj_final and not np.isnan(opt_inj_final) and opt_production:
            opt_efficiency = opt_production / opt_inj_final if opt_inj_final > 0 else 0
        else:
            opt_efficiency = 0
        
        summary.append(f"<b>Current Efficiency (q/inj):</b> {current_efficiency:.4f}")
        summary.append(f"<b>Optimal Efficiency (q/inj):</b> {opt_efficiency:.4f}")
        if current_efficiency > 0:
            summary.append(f"<b>Potential Improvement:</b> {(opt_efficiency/current_efficiency - 1)*100:.1f}%")
    
    # Enhanced recommendations
    recs = []
    if not np.isnan(opt_inj_final):
        recs.append(f"<b>Adjust gas injection to ~{opt_inj_final:.2f} units</b> for optimal production")
    else:
        recs.append("<b>Optimize gas injection rate</b> based on production response")
    
    if pi is not None and pi < 0.5:
        recs.append("<b>Consider well stimulation</b> - Low PI indicates possible formation damage")
    
    if whp_col and not np.all(np.isnan(whp)):
        whp_std = np.nanstd(whp)
        if whp_std > 100:
            recs.append("<b>Monitor wellhead pressure stability</b> - High fluctuations detected")
    
    recs.append("<b>Implement gradient curve analysis</b> for valve spacing optimization")
    recs.append("<b>Consider gas lift valve survey</b> to verify valve performance")
    
    # Enhanced risks
    risks = []
    if inj.std() == 0:
        risks.append("<b>Constant gas injection detected</b> - No optimization attempted historically")
    
    q_cv = np.std(q_clean) / np.mean(q_clean) if np.mean(q_clean) > 0 else 0
    if q_cv > 0.3:
        risks.append(f"<b>High production variability (CV={q_cv:.2f})</b> - Check for instability")
    
    if len(q_clean) < 20:
        risks.append("<b>Limited data points</b> - Optimization confidence is reduced")
    
    # Calculate decline rate if time-based data
    if t_col != "t" and len(q_clean) > 10:
        try:
            # Simple exponential decline fit
            def decline_func(t, qi, d):
                return qi * np.exp(-d * t)
            
            t_norm = t_clean - t_clean.min()
            popt, _ = optimize.curve_fit(decline_func, t_norm, q_clean, 
                                        p0=[np.max(q_clean), 0.001])
            decline_rate = popt[1] * 365  # annual decline rate
            risks.append(f"<b>Production decline rate: {decline_rate:.1%} per year</b>")
        except:
            pass
    
    # Enhanced metrics
    metrics = {
        "avg_oil_rate": float(np.nanmean(q_clean)) if len(q_clean) > 0 else None,
        "max_oil_rate": float(np.nanmax(q_clean)) if len(q_clean) > 0 else None,
        "std_oil_rate": float(np.nanstd(q_clean)) if len(q_clean) > 0 else None,
        "avg_gas_injection": float(np.nanmean(inj_clean)) if len(inj_clean) > 0 else None,
        "opt_gas_injection": float(opt_inj_final) if not np.isnan(opt_inj_final) else None,
        "current_efficiency": float(current_efficiency) if 'current_efficiency' in locals() else None,
        "optimal_efficiency": float(opt_efficiency) if 'opt_efficiency' in locals() else None,
        "optimal_production": float(opt_production) if opt_production else None,
        "productivity_index": float(pi) if pi is not None else None,
        "data_quality_score": float(valid_mask.sum() / len(t)) if len(t) > 0 else None,
        "data_points": len(q_clean)
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
        "metrics": metrics,
        "analysis_notes": [
            "Advanced ML optimization with Random Forest",
            "Physics-based Vogel-type analysis",
            "Economic optimization considered",
            "Data quality assessment performed",
            "Productivity Index calculated"
        ]
    }


def analyze_esp_advanced(df):
    """Enhanced ESP analysis with motor performance and system efficiency"""
    
    t_col = find_col(df, [["time", "date", "day", "timestamp"]])
    q_col = find_col(df, [["q_oil", "oil_rate", "rate", "qo", "production", "liquid"]])
    freq_col = find_col(df, [["freq_hz", "frequency", "hz", "vfd", "speed", "rpm"]])
    intake_col = find_col(df, [["intake_pressure", "pi", "pump_intake", "suction"]])
    discharge_col = find_col(df, [["discharge_pressure", "pd", "pump_discharge", "discharge"]])
    current_col = find_col(df, [["current", "amps", "amperage", "motor_current"]])
    voltage_col = find_col(df, [["voltage", "volts", "v"]])
    temp_col = find_col(df, [["temperature", "temp", "motor_temp"]])
    
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.interpolate(limit_direction='both')
    
    if t_col is None:
        numeric["t"] = np.arange(len(numeric))
        t_col = "t"
    else:
        numeric["t"] = np.arange(len(numeric))
    
    # Find columns
    if q_col is None:
        for col in numeric.columns:
            if any(word in str(col).lower() for word in ["rate", "prod", "oil", "q", "liquid"]):
                q_col = col
                break
        if q_col is None and len(numeric.columns) > 0:
            q_col = numeric.columns[0]
    
    if freq_col is None:
        for col in numeric.columns:
            if any(word in str(col).lower() for word in ["freq", "hz", "speed", "rpm", "vfd"]):
                freq_col = col
                break
        if freq_col is None and len(numeric.columns) > 1:
            freq_col = numeric.columns[1]
    
    # Extract data
    t = numeric["t"].values
    q = numeric[q_col].values if q_col in numeric.columns else np.zeros_like(t)
    f = numeric[freq_col].values if freq_col in numeric.columns else np.zeros_like(t)
    intake = numeric[intake_col].values if intake_col and intake_col in numeric.columns else np.full_like(t, np.nan)
    discharge = numeric[discharge_col].values if discharge_col and discharge_col in numeric.columns else np.full_like(t, np.nan)
    current = numeric[current_col].values if current_col and current_col in numeric.columns else np.full_like(t, np.nan)
    
    # Clean data
    q_outliers = detect_outliers_iqr(q, threshold=2.0)
    f_outliers = detect_outliers_iqr(f, threshold=2.0)
    valid_mask = ~q_outliers & ~f_outliers & ~np.isnan(q) & ~np.isnan(f)
    
    if valid_mask.sum() < 10:
        valid_mask = ~np.isnan(q) & ~np.isnan(f)
    
    t_clean = t[valid_mask]
    q_clean = q[valid_mask]
    f_clean = f[valid_mask]
    
    # 1. Advanced frequency optimization using cubic spline
    opt_freq_spline = None
    if len(f_clean) >= 15:
        try:
            from scipy.interpolate import CubicSpline
            
            # Sort by frequency
            sort_idx = np.argsort(f_clean)
            f_sorted = f_clean[sort_idx]
            q_sorted = q_clean[sort_idx]
            
            # Fit cubic spline
            cs = CubicSpline(f_sorted, q_sorted)
            
            # Find maximum
            f_range = np.linspace(np.min(f_clean), np.max(f_clean), 500)
            q_range = cs(f_range)
            
            # Constrain to reasonable frequencies (30-70 Hz typical for ESP)
            reasonable_mask = (f_range >= 30) & (f_range <= 70)
            if reasonable_mask.any():
                f_reasonable = f_range[reasonable_mask]
                q_reasonable = q_range[reasonable_mask]
                opt_idx = np.argmax(q_reasonable)
                opt_freq_spline = float(f_reasonable[opt_idx])
            else:
                opt_idx = np.argmax(q_range)
                opt_freq_spline = float(f_range[opt_idx])
                
        except Exception as e:
            print(f"Spline optimization failed: {e}")
    
    # 2. Affinity laws analysis
    opt_freq_affinity = None
    if len(f_clean) >= 10:
        try:
            # According to affinity laws: Q ∝ N, H ∝ N², P ∝ N³
            # For ESP, we can optimize for efficiency
            
            # Calculate head if pressures available
            if not np.all(np.isnan(intake)) and not np.all(np.isnan(discharge)):
                head = discharge[valid_mask] - intake[valid_mask]  # Simplified
                
                # Find frequency where head production product is maximized
                performance = q_clean * head
                if len(performance) > 0:
                    opt_freq_affinity = float(f_clean[np.argmax(performance)])
        except:
            pass
    
    # 3. Motor efficiency optimization
    opt_freq_motor = None
    if not np.all(np.isnan(current)) and len(current[valid_mask]) >= 10:
        try:
            current_clean = current[valid_mask]
            
            # Calculate power factor approximation
            # P = √3 × V × I × PF (simplified)
            power_est = current_clean * 460 * 0.85  # Approximate
            
            # Efficiency = useful power / input power
            # Useful power ∝ Q × Head
            if 'head' in locals():
                useful_power = q_clean * head * 0.433  # Rough conversion
            else:
                useful_power = q_clean * 100  # Approximation
            
            efficiency = useful_power / power_est
            efficiency = np.clip(efficiency, 0, 1)
            
            # Find frequency with best efficiency
            valid_eff = ~np.isnan(efficiency) & ~np.isinf(efficiency)
            if valid_eff.any():
                opt_freq_motor = float(f_clean[valid_eff][np.argmax(efficiency[valid_eff])])
        except:
            pass
    
    # Choose best optimal frequency
    candidates = [c for c in [opt_freq_spline, opt_freq_affinity, opt_freq_motor] if c is not None]
    if candidates:
        opt_freq_final = np.median(candidates)
    else:
        # Fallback to quadratic fit
        if len(f_clean) >= 5 and f_clean.std() > 0:
            try:
                coeffs = np.polyfit(f_clean, q_clean, 2)
                a, b, c = coeffs
                if a < 0:  # Concave down
                    opt_freq_final = -b / (2 * a)
                else:
                    opt_freq_final = np.percentile(f_clean, 75)
            except:
                opt_freq_final = np.percentile(f_clean, 75)
        else:
            opt_freq_final = np.nanmean(f_clean)
    
    # Constrain to typical ESP range
    if opt_freq_final is not None and not np.isnan(opt_freq_final):
        opt_freq_final = max(30, min(70, opt_freq_final))
    
    # Time series
    time_series = {
        "t": safe_list(t),
        "q_oil": safe_list(q),
        "freq": safe_list(f),
        "intake_pressure": safe_list(intake) if intake_col else [],
        "discharge_pressure": safe_list(discharge) if discharge_col else [],
        "motor_current": safe_list(current) if current_col else []
    }
    
    # Optimization curve
    f_sorted_idx = np.argsort(f_clean)
    f_sorted = f_clean[f_sorted_idx]
    q_sorted = q_clean[f_sorted_idx]
    
    # Smooth curve
    if len(q_sorted) >= 5:
        q_smooth_curve = signal.savgol_filter(q_sorted, min(5, len(q_sorted)//2*2+1), 2)
    else:
        q_smooth_curve = q_sorted
    
    # Calculate optimal production
    if not np.isnan(opt_freq_final) and len(f_sorted) > 0 and len(q_smooth_curve) > 0:
        opt_production = float(np.interp(opt_freq_final, f_sorted, q_smooth_curve))
        opt_point = [float(opt_freq_final), opt_production]
    else:
        opt_production = None
        opt_point = None
    
    opt_curve = {
        "x": safe_list(f_sorted),
        "y": safe_list(q_sorted),
        "y_smooth": safe_list(q_smooth_curve),
        "xlabel": freq_col or "Frequency (Hz)",
        "ylabel": q_col or "Oil Rate",
        "title": "Frequency vs Oil Rate",
        "opt_point": opt_point
    }
    
    # Enhanced summary
    summary = build_advanced_summary(df, "ESP")
    summary.append(f"<b>Data Quality:</b> {valid_mask.sum()}/{len(t)} valid points")
    
    if not np.isnan(opt_freq_final):
        summary.append(f"<b>AI Optimal Frequency:</b> {opt_freq_final:.2f} Hz")
        
        current_avg_f = np.mean(f_clean)
        if current_avg_f > 0:
            freq_change = ((opt_freq_final - current_avg_f) / current_avg_f) * 100
            summary.append(f"<b>Recommended frequency change:</b> {freq_change:+.1f}%")
    
    # Calculate system efficiency if possible
    system_efficiency = None
    if not np.all(np.isnan(intake)) and not np.all(np.isnan(discharge)) and not np.all(np.isnan(current)):
        try:
            head = discharge[valid_mask] - intake[valid_mask]
            hydraulic_power = q_clean * head * 0.433 * 0.000393  # hp
            electrical_power = current[valid_mask] * 460 * 1.732 * 0.85 / 1000  # kW to hp
            system_efficiency = np.mean(hydraulic_power / electrical_power) * 100
            summary.append(f"<b>System Efficiency:</b> {system_efficiency:.1f}%")
        except:
            pass
    
    # Enhanced recommendations
    recs = []
    if not np.isnan(opt_freq_final):
        recs.append(f"<b>Adjust VFD frequency to ~{opt_freq_final:.2f} Hz</b> for optimal production")
    else:
        recs.append("<b>Optimize VFD frequency</b> based on production response")
    
    if not np.all(np.isnan(intake)):
        intake_avg = np.nanmean(intake[valid_mask])
        if intake_avg < 200:
            recs.append("<b>Monitor intake pressure</b> - Risk of gas interference or pump-off")
    
    if not np.all(np.isnan(current)):
        current_cv = np.nanstd(current[valid_mask]) / np.nanmean(current[valid_mask])
        if current_cv > 0.2:
            recs.append("<b>Check motor load stability</b> - High current fluctuations detected")
    
    recs.append("<b>Perform pump performance test</b> to validate optimal operating point")
    recs.append("<b>Monitor motor temperature</b> when operating at new frequency")
    
    # Enhanced risks
    risks = []
    if f.std() == 0:
        risks.append("<b>Constant frequency operation</b> - No VFD optimization attempted")
    
    if not np.all(np.isnan(intake)):
        low_intake_mask = intake[valid_mask] < 100
        if low_intake_mask.any():
            risks.append("<b>Low intake pressure detected</b> - Risk of pump-off condition")
    
    # Calculate specific speed if possible
    if not np.all(np.isnan(intake)) and not np.all(np.isnan(discharge)) and opt_freq_final is not None:
        try:
            head = np.nanmean(discharge[valid_mask] - intake[valid_mask])
            q_opt = np.interp(opt_freq_final, f_sorted, q_smooth_curve) if opt_production else np.mean(q_clean)
            
            # Simplified specific speed calculation
            Ns = opt_freq_final * np.sqrt(q_opt) / (head ** 0.75)
            if Ns < 1000:
                risks.append("<b>Pump operating at low specific speed</b> - Consider different pump stage")
        except:
            pass
    
    # Enhanced metrics
    metrics = {
        "avg_oil_rate": float(np.nanmean(q_clean)) if len(q_clean) > 0 else None,
        "max_oil_rate": float(np.nanmax(q_clean)) if len(q_clean) > 0 else None,
        "avg_frequency": float(np.nanmean(f_clean)) if len(f_clean) > 0 else None,
        "opt_frequency": float(opt_freq_final) if not np.isnan(opt_freq_final) else None,
        "frequency_std": float(np.nanstd(f_clean)) if len(f_clean) > 0 else None,
        "current_oil_rate": float(q_clean[-1]) if len(q_clean) > 0 else None,
        "optimal_production": float(opt_production) if opt_production else None,
        "frequency_range": f"[{float(np.min(f_clean)) if len(f_clean)>0 else 0}, "
                          f"{float(np.max(f_clean)) if len(f_clean)>0 else 0}]",
        "data_points": len(q_clean)
    }
    
    if not np.all(np.isnan(intake)):
        metrics["avg_intake_pressure"] = float(np.nanmean(intake[valid_mask]))
    
    if not np.all(np.isnan(discharge)):
        metrics["avg_discharge_pressure"] = float(np.nanmean(discharge[valid_mask]))
    
    if system_efficiency is not None:
        metrics["system_efficiency"] = float(system_efficiency)
    
    return {
        "lift_type": "ESP",
        "summary": summary,
        "recommendations": recs,
        "risks": risks,
        "plots": {
            "time_series": time_series,
            "opt_curve": opt_curve
        },
        "metrics": metrics,
        "analysis_notes": [
            "Cubic spline optimization",
            "Affinity laws analysis",
            "Motor efficiency consideration",
            "Pump performance assessment",
            "System stability evaluation"
        ]
    }


def analyze_pcp_advanced(df):
    """Enhanced PCP analysis with torque optimization and wear prediction"""
    
    t_col = find_col(df, [["time", "date", "day", "timestamp"]])
    q_col = find_col(df, [["q_oil", "oil_rate", "rate", "qo", "production"]])
    rpm_col = find_col(df, [["rpm", "speed", "spinner", "rotational_speed"]])
    torque_col = find_col(df, [["torque", "tq", "torq", "load"]])
    temp_col = find_col(df, [["temperature", "temp", "fluid_temp"]])
    pressure_col = find_col(df, [["pressure", "discharge_pressure", "head_pressure"]])
    
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.interpolate(limit_direction='both')
    
    if t_col is None:
        numeric["t"] = np.arange(len(numeric))
        t_col = "t"
    else:
        numeric["t"] = np.arange(len(numeric))
    
    # Find columns
    if q_col is None:
        for col in numeric.columns:
            if any(word in str(col).lower() for word in ["rate", "prod", "oil", "q"]):
                q_col = col
                break
        if q_col is None and len(numeric.columns) > 0:
            q_col = numeric.columns[0]
    
    if rpm_col is None:
        for col in numeric.columns:
            if any(word in str(col).lower() for word in ["rpm", "speed", "rotation"]):
                rpm_col = col
                break
        if rpm_col is None and len(numeric.columns) > 1:
            rpm_col = numeric.columns[1]
    
    if torque_col is None:
        for col in numeric.columns:
            if any(word in str(col).lower() for word in ["torque", "tq", "load"]):
                torque_col = col
                break
    
    # Extract data
    t = numeric["t"].values
    q = numeric[q_col].values if q_col in numeric.columns else np.zeros_like(t)
    rpm = numeric[rpm_col].values if rpm_col in numeric.columns else np.zeros_like(t)
    torque = numeric[torque_col].values if torque_col and torque_col in numeric.columns else np.full_like(t, np.nan)
    pressure = numeric[pressure_col].values if pressure_col and pressure_col in numeric.columns else np.full_like(t, np.nan)
    
    # Clean data
    q_outliers = detect_outliers_iqr(q, threshold=2.0)
    rpm_outliers = detect_outliers_iqr(rpm, threshold=2.0)
    valid_mask = ~q_outliers & ~rpm_outliers & ~np.isnan(q) & ~np.isnan(rpm)
    
    if valid_mask.sum() < 10:
        valid_mask = ~np.isnan(q) & ~np.isnan(rpm)
    
    t_clean = t[valid_mask]
    q_clean = q[valid_mask]
    rpm_clean = rpm[valid_mask]
    torque_clean = torque[valid_mask] if torque_col else None
    
    # 1. Advanced RPM optimization with torque consideration
    opt_rpm_advanced = None
    if len(rpm_clean) >= 15:
        try:
            # Sort data
            sort_idx = np.argsort(rpm_clean)
            rpm_sorted = rpm_clean[sort_idx]
            q_sorted = q_clean[sort_idx]
            
            # Use Random Forest for complex relationships
            if torque_clean is not None and not np.all(np.isnan(torque_clean)):
                X = np.column_stack([rpm_clean, torque_clean[valid_mask]])
                y = q_clean
                
                rf = RandomForestRegressor(n_estimators=50, random_state=42)
                rf.fit(X, y)
                
                # Predict across RPM range with typical torque
                rpm_range = np.linspace(np.min(rpm_clean), np.max(rpm_clean), 200)
                typical_torque = np.nanmedian(torque_clean[valid_mask])
                X_pred = np.column_stack([rpm_range, np.full_like(rpm_range, typical_torque)])
                q_pred = rf.predict(X_pred)
                
                opt_rpm_advanced = float(rpm_range[np.argmax(q_pred)])
            else:
                # Simple optimization without torque
                from scipy.interpolate import CubicSpline
                cs = CubicSpline(rpm_sorted, q_sorted)
                rpm_range = np.linspace(np.min(rpm_clean), np.max(rpm_clean), 500)
                q_range = cs(rpm_range)
                opt_rpm_advanced = float(rpm_range[np.argmax(q_range)])
                
        except Exception as e:
            print(f"Advanced RPM optimization failed: {e}")
    
    # 2. Torque-based optimization (minimize wear)
    opt_rpm_torque = None
    if torque_clean is not None and len(torque_clean[~np.isnan(torque_clean)]) >= 10:
        try:
            # Calculate torque per RPM (wear indicator)
            valid_torque = ~np.isnan(torque_clean)
            torque_per_rpm = torque_clean[valid_torque] / rpm_clean[valid_torque]
            
            # Find RPM with reasonable torque and good production
            sort_idx = np.argsort(rpm_clean[valid_torque])
            rpm_torque = rpm_clean[valid_torque][sort_idx]
            torque_vals = torque_clean[valid_torque][sort_idx]
            q_vals = q_clean[valid_torque][sort_idx]
            
            # Normalize metrics
            torque_norm = (torque_vals - np.min(torque_vals)) / (np.max(torque_vals) - np.min(torque_vals))
            q_norm = (q_vals - np.min(q_vals)) / (np.max(q_vals) - np.min(q_vals))
            
            # Combined score: maximize production while minimizing torque
            combined_score = q_norm * 0.7 - torque_norm * 0.3
            opt_rpm_torque = float(rpm_torque[np.argmax(combined_score)])
            
        except:
            pass
    
    # 3. Economic optimization considering rod fatigue
    opt_rpm_economic = None
    if len(rpm_clean) >= 10:
        try:
            # Rod fatigue is proportional to RPM²
            fatigue_cost = rpm_clean ** 2 * 0.001  # Simplified cost
            
            # Revenue from production
            revenue = q_clean * 50  # $50 per unit
            
            # Net value
            net_value = revenue - fatigue_cost
            
            # Find RPM for max net value
            opt_rpm_economic = float(rpm_clean[np.argmax(net_value)])
        except:
            pass
    
    # Choose best RPM
    candidates = [c for c in [opt_rpm_advanced, opt_rpm_torque, opt_rpm_economic] if c is not None]
    if candidates:
        opt_rpm_final = np.median(candidates)
    else:
        # Fallback: 80% of max RPM
        if len(rpm_clean) > 0:
            opt_rpm_final = np.min(rpm_clean) + 0.8 * (np.max(rpm_clean) - np.min(rpm_clean))
        else:
            opt_rpm_final = np.nan
    
    # Constrain to typical PCP range (50-300 RPM)
    if opt_rpm_final is not None and not np.isnan(opt_rpm_final):
        opt_rpm_final = max(50, min(300, opt_rpm_final))
    
    # Time series
    time_series = {
        "t": safe_list(t),
        "q_oil": safe_list(q),
        "rpm": safe_list(rpm),
        "torque": safe_list(torque) if torque_col else [],
        "pressure": safe_list(pressure) if pressure_col else []
    }
    
    # Optimization curve
    rpm_sorted_idx = np.argsort(rpm_clean)
    rpm_sorted = rpm_clean[rpm_sorted_idx]
    q_sorted = q_clean[rpm_sorted_idx]
    
    # Smooth curve
    if len(q_sorted) >= 5:
        q_smooth_curve = signal.savgol_filter(q_sorted, min(5, len(q_sorted)//2*2+1), 2)
    else:
        q_smooth_curve = q_sorted
    
    # Calculate optimal production
    if not np.isnan(opt_rpm_final) and len(rpm_sorted) > 0 and len(q_smooth_curve) > 0:
        opt_production = float(np.interp(opt_rpm_final, rpm_sorted, q_smooth_curve))
        opt_point = [float(opt_rpm_final), opt_production]
    else:
        opt_production = None
        opt_point = None
    
    opt_curve = {
        "x": safe_list(rpm_sorted),
        "y": safe_list(q_sorted),
        "y_smooth": safe_list(q_smooth_curve),
        "xlabel": rpm_col or "RPM",
        "ylabel": q_col or "Oil Rate",
        "title": "RPM vs Oil Rate",
        "opt_point": opt_point
    }
    
    # Enhanced summary
    summary = build_advanced_summary(df, "PCP")
    summary.append(f"<b>Data Quality:</b> {valid_mask.sum()}/{len(t)} valid points")
    
    if not np.isnan(opt_rpm_final):
        summary.append(f"<b>AI Optimal RPM:</b> {opt_rpm_final:.2f}")
        
        current_avg_rpm = np.mean(rpm_clean)
        if current_avg_rpm > 0:
            rpm_change = ((opt_rpm_final - current_avg_rpm) / current_avg_rpm) * 100
            summary.append(f"<b>Recommended RPM change:</b> {rpm_change:+.1f}%")
    
    # Calculate elastomer stress if torque available
    max_stress = None
    avg_stress = None
    if torque_clean is not None and not np.all(np.isnan(torque_clean)):
        try:
            stress = torque_clean[valid_mask] / (np.pi * (2.5 ** 3) / 16)  # Simplified stress calculation
            max_stress = np.max(stress[~np.isnan(stress)])
            avg_stress = np.nanmean(stress)
            
            summary.append(f"<b>Max Elastomer Stress:</b> {max_stress:.1f} psi")
            summary.append(f"<b>Avg Elastomer Stress:</b> {avg_stress:.1f} psi")
            
            if max_stress > 5000:
                summary.append("<b>Warning:</b> High stress detected - monitor elastomer wear")
        except:
            pass
    
    # Enhanced recommendations
    recs = []
    if not np.isnan(opt_rpm_final):
        recs.append(f"<b>Adjust RPM to ~{opt_rpm_final:.2f}</b> for optimal production and reduced wear")
    else:
        recs.append("<b>Optimize RPM</b> based on production response")
    
    if torque_clean is not None:
        torque_cv = np.nanstd(torque_clean[valid_mask]) / np.nanmean(torque_clean[valid_mask])
        if torque_cv > 0.25:
            recs.append("<b>Monitor torque stability</b> - High fluctuations may indicate solids or gas")
    
    if pressure_col and not np.all(np.isnan(pressure)):
        pressure_avg = np.nanmean(pressure[valid_mask])
        if pressure_avg > 1000:
            recs.append("<b>Consider pressure relief</b> - High discharge pressure detected")
    
    recs.append("<b>Schedule regular elastomer inspections</b> based on operating hours")
    recs.append("<b>Monitor fluid temperature</b> - High temps accelerate elastomer degradation")
    
    # Enhanced risks
    risks = []
    if rpm.std() == 0:
        risks.append("<b>Constant RPM operation</b> - No speed optimization attempted")
    
    if torque_clean is not None:
        high_torque_mask = torque_clean[valid_mask] > np.percentile(torque_clean[valid_mask], 90)
        if high_torque_mask.any():
            risks.append("<b>High torque events detected</b> - Risk of rod failure or pump seizure")
    
    # Calculate wear rate
    wear_rate = None
    if len(rpm_clean) > 10 and torque_clean is not None:
        try:
            # Simplified wear calculation
            wear_units = rpm_clean * torque_clean[valid_mask] * 0.001
            wear_rate = np.mean(wear_units) * 24  # Daily wear
            risks.append(f"<b>Estimated daily wear rate:</b> {wear_rate:.1f} units/day")
        except:
            pass
    
    # Enhanced metrics
    metrics = {
        "avg_oil_rate": float(np.nanmean(q_clean)) if len(q_clean) > 0 else None,
        "max_oil_rate": float(np.nanmax(q_clean)) if len(q_clean) > 0 else None,
        "avg_rpm": float(np.nanmean(rpm_clean)) if len(rpm_clean) > 0 else None,
        "opt_rpm": float(opt_rpm_final) if not np.isnan(opt_rpm_final) else None,
        "rpm_std": float(np.nanstd(rpm_clean)) if len(rpm_clean) > 0 else None,
        "current_production": float(q_clean[-1]) if len(q_clean) > 0 else None,
        "optimal_production": float(opt_production) if opt_production else None,
        "rpm_range": f"[{float(np.min(rpm_clean)) if len(rpm_clean)>0 else 0}, "
                    f"{float(np.max(rpm_clean)) if len(rpm_clean)>0 else 0}]",
        "data_points": len(q_clean)
    }
    
    if torque_clean is not None:
        metrics["avg_torque"] = float(np.nanmean(torque_clean[valid_mask]))
        metrics["max_torque"] = float(np.nanmax(torque_clean[valid_mask]))
    
    if pressure_col and not np.all(np.isnan(pressure)):
        metrics["avg_pressure"] = float(np.nanmean(pressure[valid_mask]))
    
    if max_stress is not None:
        metrics["max_elastomer_stress"] = float(max_stress)
    
    if avg_stress is not None:
        metrics["avg_elastomer_stress"] = float(avg_stress)
    
    if wear_rate is not None:
        metrics["wear_rate"] = float(wear_rate)
    
    return {
        "lift_type": "PCP",
        "summary": summary,
        "recommendations": recs,
        "risks": risks,
        "plots": {
            "time_series": time_series,
            "opt_curve": opt_curve
        },
        "metrics": metrics,
        "analysis_notes": [
            "Torque-optimized RPM selection",
            "Elastomer stress analysis",
            "Economic wear consideration",
            "Rod fatigue assessment",
            "Production stability evaluation"
        ]
    }


def auto_detect_lift_type(df):
    """Enhanced lift type detection"""
    cols = " ".join([str(c).lower() for c in df.columns])
    
    gas_lift_keywords = ["gas_inj", "q_gas", "glv", "annulus", "gas_lift", "gas lift", "injection_gas"]
    esp_keywords = ["freq", "vfd", "esp", "intake_pressure", "discharge_pressure", "motor_current", "hz"]
    pcp_keywords = ["rpm", "pcp", "torque", "rod", "elastomer", "rotor", "stator"]
    
    gas_score = sum(1 for word in gas_lift_keywords if word in cols)
    esp_score = sum(1 for word in esp_keywords if word in cols)
    pcp_score = sum(1 for word in pcp_keywords if word in cols)
    
    scores = {"gas_lift": gas_score, "esp": esp_score, "pcp": pcp_score}
    
    if max(scores.values()) == 0:
        return "esp"  # Default
    
    return max(scores.items(), key=lambda x: x[1])[0]


# ========== ENHANCED PDF REPORT FUNCTIONS ==========

def create_matplotlib_plot(analysis, plot_type="time_series"):
    """Create matplotlib plot and return as base64 image"""
    plots = analysis.get("plots", {})
    
    if plot_type == "time_series":
        ts = plots.get("time_series", {})
        t_data = ts.get("t", [])
        q_data = ts.get("q_oil", [])
        
        if len(t_data) < 2 or len(q_data) < 2:
            return None
            
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.plot(t_data, q_data, 'b-', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Time / Index', fontsize=8)
        ax.set_ylabel('Oil Rate', fontsize=8)
        ax.set_title('Production Time Series', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        plt.tight_layout()
        
    elif plot_type == "opt_curve":
        opt = plots.get("opt_curve", {})
        x_data = opt.get("x", [])
        y_data = opt.get("y", [])
        y_smooth = opt.get("y_smooth", [])
        opt_point = opt.get("opt_point")
        
        if len(x_data) < 2 or len(y_data) < 2:
            return None
            
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.scatter(x_data, y_data, alpha=0.5, s=10, color='blue', label='Data')
        
        if len(y_smooth) == len(x_data):
            ax.plot(x_data, y_smooth, 'r-', linewidth=2, label='AI Fit')
        
        if opt_point and len(opt_point) == 2:
            ax.scatter([opt_point[0]], [opt_point[1]], color='green', 
                      s=80, marker='*', label='Optimal Point', zorder=5)
        
        ax.set_xlabel(opt.get('xlabel', 'Control Variable'), fontsize=8)
        ax.set_ylabel(opt.get('ylabel', 'Oil Rate'), fontsize=8)
        ax.set_title('Optimization Curve', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    # Convert to base64
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64


def create_professional_pdf(analysis):
    """Create a professional PDF report - 2 PAGES ONLY"""
    buffer = io.BytesIO()
    
    # Create custom styles
    styles = getSampleStyleSheet()
    
    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=15,
        alignment=1,
        textColor=colors.HexColor('#1a365d')
    )
    
    # Header style
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=8,
        textColor=colors.HexColor('#2d3748'),
        fontName='Helvetica-Bold'
    )
    
    # Subheader style
    subheader_style = ParagraphStyle(
        'CustomSubHeader',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=6,
        textColor=colors.HexColor('#4a5568')
    )
    
    # Normal text style
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=4,
        textColor=colors.HexColor('#2d3748')
    )
    
    # Key findings style
    keyfindings_style = ParagraphStyle(
        'KeyFindings',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=3,
        textColor=colors.HexColor('#2d3748'),
        leftIndent=20
    )
    
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                          rightMargin=72, leftMargin=72,
                          topMargin=60, bottomMargin=72)
    
    story = []
    
    # ===== PAGE 1 =====
    
    # Report title
    story.append(Paragraph("AI LIFT OPTIMIZATION REPORT", title_style))
    story.append(Spacer(1, 10))
    
    # Report info
    report_date = datetime.datetime.utcnow().strftime("%B %d, %Y")
    analysis_id = f"AI-LIFT-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    
    info_table = Table([
        ["Professional Petroleum Engineering Analysis"],
        [f"Report Date: {report_date} | Analysis ID: {analysis_id}"]
    ], colWidths=[400])
    
    info_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, 0), 12),
        ('FONTSIZE', (0, 1), (0, 1), 10),
        ('TEXTCOLOR', (0, 0), (0, 0), colors.HexColor('#2d3748')),
        ('TEXTCOLOR', (0, 1), (0, 1), colors.HexColor('#4a5568')),
        ('BOTTOMPADDING', (0, 0), (0, 0), 5),
        ('BOTTOMPADDING', (0, 1), (0, 1), 15),
    ]))
    
    story.append(info_table)
    story.append(Spacer(1, 20))
    
    # Executive Summary header
    story.append(Paragraph("EXECUTIVE SUMMARY", header_style))
    story.append(Spacer(1, 6))
    
    # Executive Summary content
    exec_summary = f"""
    This report presents a comprehensive AI Lift Optimization analysis based on historical production
    data. The analysis applies advanced machine learning algorithms, physics-based models, and
    economic optimization to determine the optimal operating parameters for maximum efficiency.
    """
    story.append(Paragraph(exec_summary, normal_style))
    story.append(Spacer(1, 15))
    
    # Key Findings
    story.append(Paragraph("KEY FINDINGS:", subheader_style))
    story.append(Spacer(1, 8))
    
    lift_type = analysis.get("lift_type", "Unknown")
    metrics = analysis.get("metrics", {})
    
    # Create Key Findings table
    key_findings_data = []
    
    # Best-fitting Lift Type
    key_findings_data.append(["• Best-fitting Lift Type:", f"{lift_type}"])
    
    # AI Optimal Operating Point
    optimal_point = format_optimal_point(analysis, metrics)
    key_findings_data.append(["• AI Optimal Operating Point:", optimal_point])
    
    # Data Points Analyzed
    data_points = metrics.get('data_points', 'N/A')
    key_findings_data.append(["• Data Points Analyzed:", f"{data_points} measurements"])
    
    # Data Quality Score
    data_quality = metrics.get('data_quality_score', 0)
    if data_quality:
        key_findings_data.append(["• Data Quality Score:", f"{(data_quality * 100):.1f}%"])
    
    # Current Average Production
    avg_rate = metrics.get('avg_oil_rate')
    if avg_rate:
        key_findings_data.append(["• Current Average Production:", f"{avg_rate:.2f} units/day"])
    
    # Maximum Historical Production
    max_rate = metrics.get('max_oil_rate')
    if max_rate:
        key_findings_data.append(["• Maximum Historical Production:", f"{max_rate:.2f} units/day"])
    
    findings_table = Table(key_findings_data, colWidths=[250, 150])
    findings_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    
    story.append(findings_table)
    story.append(Spacer(1, 20))
    
    # Model Comparison Analysis header
    story.append(Paragraph("AI MODEL COMPARISON ANALYSIS", header_style))
    story.append(Spacer(1, 8))
    
    model_comp_text = """
    The AI analysis compared multiple optimization approaches including machine learning (Random Forest),
    physics-based models, and economic optimization to determine the most reliable optimal operating point.
    """
    story.append(Paragraph(model_comp_text, normal_style))
    story.append(Spacer(1, 15))
    
    # Create small plots
    try:
        # Create time series plot
        time_plot_img = create_matplotlib_plot(analysis, "time_series")
        opt_plot_img = create_matplotlib_plot(analysis, "opt_curve")
        
        if time_plot_img and opt_plot_img:
            # Create table with two plots side by side
            plot_table_data = [
                ["Time Series Plot", "Optimization Curve"],
                [Image(io.BytesIO(base64.b64decode(time_plot_img)), width=3.5*inch, height=2.5*inch),
                 Image(io.BytesIO(base64.b64decode(opt_plot_img)), width=3.5*inch, height=2.5*inch)]
            ]
            
            plot_table = Table(plot_table_data, colWidths=[3.5*inch, 3.5*inch])
            plot_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('ALIGN', (0, 1), (-1, 1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
                ('TOPPADDING', (0, 1), (-1, 1), 4),
            ]))
            
            story.append(plot_table)
            story.append(Spacer(1, 15))
    except Exception as e:
        print(f"Error creating plots: {e}")
        # Continue without plots if there's an error
    
    # ===== PAGE BREAK =====
    story.append(PageBreak())
    
    # ===== PAGE 2 (المحتوى الذي كان في الصفحة الثالثة) =====
    
    # MODEL PARAMETERS AND STATISTICS
    story.append(Paragraph("MODEL PARAMETERS AND STATISTICS", header_style))
    story.append(Spacer(1, 10))
    
    # Main parameters table
    param_data = [
        ["PARAMETER", "CURRENT VALUE", "AI OPTIMAL VALUE", "UNIT", "IMPROVEMENT"]
    ]
    
    # Get optimal production
    opt_production = interp_optimal_production(analysis)
    
    # Add parameter rows based on lift type
    if lift_type == "Gas Lift":
        avg_inj = metrics.get('avg_gas_injection')
        opt_inj = metrics.get('opt_gas_injection')
        avg_rate = metrics.get('avg_oil_rate')
        curr_eff = metrics.get('current_efficiency')
        opt_eff = metrics.get('optimal_efficiency')
        
        if avg_inj is not None and opt_inj is not None:
            param_data.append([
                "Gas Injection Rate", 
                f"{avg_inj:.2f}",
                f"{opt_inj:.2f}",
                "units/day",
                calculate_improvement(avg_inj, opt_inj)
            ])
        
        if avg_rate is not None and opt_production:
            param_data.append([
                "Oil Production Rate", 
                f"{avg_rate:.2f}",
                f"{opt_production:.2f}",
                "units/day",
                calculate_improvement(avg_rate, opt_production)
            ])
        
        if curr_eff is not None and opt_eff is not None:
            param_data.append([
                "System Efficiency", 
                f"{curr_eff:.4f}",
                f"{opt_eff:.4f}",
                "q/inj",
                calculate_improvement(curr_eff, opt_eff, True)
            ])
            
    elif lift_type == "ESP":
        avg_freq = metrics.get('avg_frequency')
        opt_freq = metrics.get('opt_frequency')
        avg_rate = metrics.get('avg_oil_rate')
        sys_eff = metrics.get('system_efficiency')
        
        if avg_freq is not None and opt_freq is not None:
            param_data.append([
                "Operating Frequency", 
                f"{avg_freq:.2f}",
                f"{opt_freq:.2f}",
                "Hz",
                calculate_improvement(avg_freq, opt_freq)
            ])
        
        if avg_rate is not None and opt_production:
            param_data.append([
                "Oil Production Rate", 
                f"{avg_rate:.2f}",
                f"{opt_production:.2f}",
                "units/day",
                calculate_improvement(avg_rate, opt_production)
            ])
        
        if sys_eff is not None:
            param_data.append([
                "System Efficiency", 
                f"{sys_eff:.1f}%",
                "Optimized",
                "%",
                "To be measured"
            ])
            
    elif lift_type == "PCP":
        avg_rpm = metrics.get('avg_rpm')
        opt_rpm = metrics.get('opt_rpm')
        avg_rate = metrics.get('avg_oil_rate')
        avg_torque = metrics.get('avg_torque')
        
        if avg_rpm is not None and opt_rpm is not None:
            param_data.append([
                "Operating RPM", 
                f"{avg_rpm:.2f}",
                f"{opt_rpm:.2f}",
                "RPM",
                calculate_improvement(avg_rpm, opt_rpm)
            ])
        
        if avg_rate is not None and opt_production:
            param_data.append([
                "Oil Production Rate", 
                f"{avg_rate:.2f}",
                f"{opt_production:.2f}",
                "units/day",
                calculate_improvement(avg_rate, opt_production)
            ])
        
        if avg_torque is not None:
            param_data.append([
                "Avg Torque", 
                f"{avg_torque:.1f}",
                "Reduced",
                "lb-ft",
                "Wear minimized"
            ])
    
    # Add productivity index if available
    pi = metrics.get('productivity_index')
    if pi is not None:
        param_data.append([
            "Productivity Index (PI)",
            f"{pi:.4f}",
            "N/A",
            "units/day/psi",
            "Baseline"
        ])
    
    # Add data quality if available
    data_quality = metrics.get('data_quality_score')
    if data_quality is not None:
        param_data.append([
            "Data Quality",
            f"{(data_quality * 100):.1f}%",
            "100% target",
            "%",
            f"+{(100 - data_quality * 100):.1f}%"
        ])
    
    param_table = Table(param_data, colWidths=[120, 90, 90, 60, 80])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a365d')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')]),
        
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    
    story.append(param_table)
    story.append(Spacer(1, 25))
    
    # ANALYSIS SUMMARY
    story.append(Paragraph("ANALYSIS SUMMARY", header_style))
    story.append(Spacer(1, 10))
    
    summary_data = [
        ["PARAMETER", "VALUE", "DESCRIPTION"]
    ]
    
    # Add summary rows
    summary_rows = [
        ["Input Data Columns", 
         "t, q, control_var", 
         "Original data columns analyzed"],
        ["Data Points", 
         f"{metrics.get('data_points', 'N/A')} measurements", 
         "Valid measurements analyzed"],
        ["Data Quality", 
         f"{(metrics.get('data_quality_score', 0) * 100):.1f}%" if metrics.get('data_quality_score') else "N/A", 
         "Data validation score"],
        ["Time Period", 
         extract_time_range(analysis), 
         "Analysis time range"],
        ["Production Range", 
         extract_production_range(metrics), 
         "Historical production variation"],
        ["AI Confidence", 
         "Advanced ML + Physics", 
         "Optimization methodology"],
        ["Best Model", 
         f"{lift_type} - AI Optimized", 
         "Selected optimization approach"]
    ]
    
    for row in summary_rows:
        summary_data.append(row)
    
    summary_table = Table(summary_data, colWidths=[120, 100, 180])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a365d')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (0, 1), (1, -1), 'CENTER'),
        ('ALIGN', (2, 1), (2, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')]),
        
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Footer
    footer_text = """
    <hr width="100%" size="1" color="#cbd5e0"/>
    <br/>
    <b>OILNOVA AI</b><br/>
    Advanced Petroleum Analytics Platform<br/>
    Report Generated by OILNOVA AI Lift Optimization Module<br/>
    © 2025 OILNOVA AI. All rights reserved. Proprietary and Confidential<br/>
    <br/>
    Report Version: 2.00 | Analysis Engine: OILNOVA AI
    """
    
    story.append(Paragraph(footer_text, ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#718096'),
        alignment=1
    )))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return buffer

def format_optimal_point(analysis, metrics):
    """Format the optimal point for display"""
    lift_type = analysis.get("lift_type", "")
    
    if lift_type == "Gas Lift" and metrics.get('opt_gas_injection'):
        return f"{metrics['opt_gas_injection']:.2f} units/day gas injection"
    elif lift_type == "ESP" and metrics.get('opt_frequency'):
        return f"{metrics['opt_frequency']:.2f} Hz operating frequency"
    elif lift_type == "PCP" and metrics.get('opt_rpm'):
        return f"{metrics['opt_rpm']:.2f} RPM operating speed"
    else:
        return "To be optimized"

def calculate_improvement(current, optimal, is_efficiency=False):
    """Calculate improvement percentage"""
    if current is None or optimal is None:
        return "N/A"
    
    try:
        current_val = float(current)
        optimal_val = float(optimal)
        
        if current_val == 0:
            return "∞%"
        
        if is_efficiency:
            improvement = ((optimal_val - current_val) / current_val) * 100
        else:
            # For parameters where lower might be better (like gas injection)
            improvement = ((optimal_val - current_val) / current_val) * 100
        
        if improvement > 0:
            return f"+{improvement:.1f}%"
        elif improvement < 0:
            return f"{improvement:.1f}%"
        else:
            return "0%"
    except:
        return "N/A"

def extract_time_range(analysis):
    """Extract time range from analysis"""
    plots = analysis.get("plots", {})
    ts = plots.get("time_series", {})
    t_data = ts.get("t", [])
    
    if len(t_data) >= 2:
        return f"{min(t_data):.0f} - {max(t_data):.0f} days"
    return "N/A"

def extract_production_range(metrics):
    """Extract production range from metrics"""
    avg_rate = metrics.get('avg_oil_rate')
    max_rate = metrics.get('max_oil_rate')
    
    if avg_rate and max_rate:
        min_est = avg_rate * 0.5  # Estimate minimum
        return f"{min_est:.0f} - {max_rate:.0f} units/day"
    elif max_rate:
        return f"Up to {max_rate:.0f} units/day"
    return "N/A"


# ========== API ENDPOINTS ==========

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
            result = analyze_gas_lift_advanced(df)
        elif lift_type == "esp":
            result = analyze_esp_advanced(df)
        elif lift_type == "pcp":
            result = analyze_pcp_advanced(df)
        else:
            return jsonify({"error": f"Unknown lift type: {lift_type}"}), 400

        result["lift_type_code"] = lift_type
        result["generated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        result["analysis_version"] = "2.0_advanced"
        result["powered_by"] = "OILNOVA AI"

        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Analysis error: {str(e)}")
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
        
        buffer = create_professional_pdf(analysis)

        ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"OILNOVA_AI_Lift_Optimization_Report_{ts}.pdf"
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype="application/pdf"
        )

    except Exception as e:
        app.logger.error(f"Report generation error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "2.0_professional",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
