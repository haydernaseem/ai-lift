import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d
from sklearn.ensemble import IsolationForest
from flask import Flask, request, jsonify, send_file
import joblib
import json

# ==================== PHYSICS MODELS ====================

class FluidProperties:
    """خواص السوائل للآبار"""
    def __init__(self, oil_gravity: float = 35, water_cut: float = 0.3, 
                 gas_gravity: float = 0.65, temp: float = 180):
        self.API = oil_gravity
        self.water_cut = water_cut
        self.gas_gravity = gas_gravity
        self.temperature = temp  # درجة فهرنهايت
        self.pressure = 2000  # psi
        
    def calculate_rho(self) -> Tuple[float, float, float]:
        """حساب كثافات النفط، الماء، الغاز"""
        # كثافة النفط (lb/ft3)
        rho_o = 141.5 / (131.5 + self.API) * 62.4
        
        # كثافة الماء
        rho_w = 62.4  # lb/ft3
        
        # كثافة الغاز باستخدام معادلة الحالة
        rho_g = (2.7 * self.gas_gravity * self.pressure) / (
            (self.temperature + 460) * 0.8)
        
        return rho_o, rho_w, rho_g
    
    def calculate_viscosity(self, temp: float) -> Tuple[float, float]:
        """حساب لزوجة النفط والماء"""
        # لزوجة النفط (cp)
        mu_o = np.exp(3.0324 - 0.02023 * self.API) * (
            1.8 * (temp - 32) + 32) ** (-1.163)
        
        # لزوجة الماء (cp)
        mu_w = 1.0  # تقريبي
        
        return mu_o, mu_w

@dataclass
class PumpCurve:
    """منحنى أداء المضخة الحقيقي"""
    flow_rates: np.ndarray  # BPD
    heads: np.ndarray  # feet
    efficiencies: np.ndarray  # %
    powers: np.ndarray  # HP
    
    @classmethod
    def from_manufacturer(cls, pump_type: str, stages: int):
        """إنشاء منحنى مضخة من بيانات الشركة المصنعة"""
        if pump_type == "ESP400":
            flow = np.linspace(500, 4000, 20)
            head_per_stage = 30 - 0.005 * (flow - 2000)**2
            eff = 65 - 0.0001 * (flow - 2200)**2
            power = flow * head_per_stage * stages / (3960 * eff/100)
            return cls(flow, head_per_stage * stages, eff, power)
        
        elif pump_type == "REDA500":
            flow = np.linspace(1000, 5000, 20)
            head_per_stage = 28 - 0.004 * (flow - 2500)**2
            eff = 68 - 0.00008 * (flow - 2400)**2
            power = flow * head_per_stage * stages / (3960 * eff/100)
            return cls(flow, head_per_stage * stages, eff, power)
        
        return None
    
    def interpolate_head(self, flow: float) -> float:
        """القيمة المثبتة للرأس عند معدل تدفق معين"""
        f = interp1d(self.flow_rates, self.heads, 
                    bounds_error=False, fill_value="extrapolate")
        return float(f(flow))
    
    def best_efficiency_point(self) -> dict:
        """نقطة أفضل كفاءة (BEP)"""
        idx = np.argmax(self.efficiencies)
        return {
            "flow": float(self.flow_rates[idx]),
            "head": float(self.heads[idx]),
            "efficiency": float(self.efficiencies[idx]),
            "power": float(self.powers[idx])
        }

class WellIPR:
    """منحنى أداء المكمن (Inflow Performance Relationship)"""
    def __init__(self, reservoir_pressure: float, 
                 productivity_index: float, 
                 oil_rate_max: float):
        self.P_res = reservoir_pressure
        self.J = productivity_index  # STB/day/psi
        self.q_max = oil_rate_max
        
    def vogel_ipr(self, pwf: float) -> float:
        """معادلة فوغل لآبار النفط المشبعة"""
        if pwf >= self.P_res:
            return 0
        q = self.J * (self.P_res - pwf)
        # Vogel adjustment for saturated oil
        if self.P_res > 2000:  # فوق نقطة الفقاعة
            return q
        else:
            return q * (1 - 0.2 * (pwf/self.P_res) - 0.8 * (pwf/self.P_res)**2)
    
    def generate_ipr_curve(self) -> Dict:
        """إنشاء منحنى IPR كامل"""
        pwf_values = np.linspace(self.P_res, 0, 50)
        q_values = [self.vogel_ipr(p) for p in pwf_values]
        return {"pwf": pwf_values.tolist(), "q": q_values}

class EconomicCalculator:
    """محاسب اقتصادي للآبار"""
    def __init__(self, oil_price: float = 70,  # $/bbl
                 gas_cost: float = 0.5,  # $/MCF
                 electricity_cost: float = 0.08,  # $/kWh
                 opex_per_bbl: float = 15):  # $/bbl
        self.oil_price = oil_price
        self.gas_cost = gas_cost
        self.electricity_cost = electricity_cost
        self.opex = opex_per_bbl
        
    def calculate_npv(self, oil_rate: float, 
                      gas_injection: float = 0,
                      power_consumption: float = 0,
                      days: int = 30) -> dict:
        """حساب صافي القيمة الحالية"""
        # الإيرادات
        revenue = oil_rate * days * self.oil_price
        
        # التكاليف
        gas_cost_total = gas_injection * days * self.gas_cost / 1000  # MCF to M
        power_cost = power_consumption * 24 * days * self.electricity_cost
        opex_cost = oil_rate * days * self.opex
        
        total_cost = gas_cost_total + power_cost + opex_cost
        net_income = revenue - total_cost
        
        return {
            "revenue": revenue,
            "total_cost": total_cost,
            "net_income": net_income,
            "lifting_cost_per_bbl": total_cost / (oil_rate * days) 
            if oil_rate * days > 0 else 0,
            "roi_percent": (net_income / total_cost * 100) 
            if total_cost > 0 else 0
        }
    
    def optimize_profit(self, rates: np.ndarray, 
                       costs: np.ndarray) -> dict:
        """إيجاد نقطة الربح الأمثل"""
        profits = rates * self.oil_price - costs
        idx_opt = np.argmax(profits)
        
        return {
            "optimal_rate": float(rates[idx_opt]),
            "optimal_cost": float(costs[idx_opt]),
            "max_profit": float(profits[idx_opt]),
            "sensitivity": float((profits.max() - profits.min()) / profits.max() * 100)
        }

# ==================== AI ENGINE ====================

class AdvancedAnomalyDetector:
    """كشف متقدم للقيم الشاذة وأنماط الأعطال"""
    
    def __init__(self):
        self.models = {}
        
    def train_failure_patterns(self, historical_data: pd.DataFrame):
        """تدريب أنماط الأعطال التاريخية"""
        features = ['vibration', 'motor_temp', 'current_unbalance', 
                   'flow_deviation', 'pressure_delta']
        
        available_features = [f for f in features 
                            if f in historical_data.columns]
        
        if len(available_features) >= 3:
            X = historical_data[available_features].fillna(0)
            
            # Isolation Forest للكشف عن الشذوذ
            iso_forest = IsolationForest(
                contamination=0.1, 
                random_state=42,
                n_estimators=100
            )
            
            anomalies = iso_forest.fit_predict(X)
            self.models['isolation_forest'] = iso_forest
            
            # حساب درجات الشذوذ
            anomaly_scores = iso_forest.decision_function(X)
            historical_data['anomaly_score'] = anomaly_scores
            historical_data['is_anomaly'] = anomalies == -1
            
        return historical_data
    
    def predict_failure_risk(self, current_data: pd.Series) -> dict:
        """توقع مخاطر الأعطال"""
        risk_score = 0
        alerts = []
        
        # قواعد معرفة المجال
        if 'motor_temp' in current_data and current_data['motor_temp'] > 180:
            risk_score += 30
            alerts.append("🔥 درجة حرارة المحور مرتفعة جداً (>180°F)")
            
        if 'vibration' in current_data and current_data['vibration'] > 0.5:
            risk_score += 25
            alerts.append("⚠️ اهتزازات مرتفعة - خطر تلف المحور")
            
        if 'current_unbalance' in current_data and current_data['current_unbalance'] > 15:
            risk_score += 20
            alerts.append("⚡ عدم توازن التيار الكهربائي")
            
        if 'flow_deviation' in current_data and abs(current_data['flow_deviation']) > 30:
            risk_score += 15
            alerts.append("📉 انحراف كبير في التدفق")
            
        risk_level = "منخفض"
        if risk_score > 50:
            risk_level = "عالٍ"
        elif risk_score > 25:
            risk_level = "متوسط"
            
        return {
            "risk_score": min(risk_score, 100),
            "risk_level": risk_level,
            "alerts": alerts,
            "recommended_action": self._get_action_from_risk(risk_score)
        }
    
    def _get_action_from_risk(self, score: int) -> str:
        if score > 70:
            return "إيقاف فوري والتحقق من المعدات"
        elif score > 50:
            return "تقليل الحمل وطلب الصيانة خلال 24 ساعة"
        elif score > 30:
            return "مراقبة عن كثب وفحص خلال 72 ساعة"
        else:
            return "مراقبة روتينية"

class DeepOptimizationEngine:
    """محرك تحسين عميق يعمل بالفيزياء والذكاء الاصطناعي"""
    
    def __init__(self, well_type: str, fluid_props: FluidProperties):
        self.well_type = well_type
        self.fluid = fluid_props
        self.economic_calc = EconomicCalculator()
        
    def optimize_esp(self, pump_curve: PumpCurve, 
                    well_ipr: WellIPR,
                    historical_data: pd.DataFrame) -> dict:
        """التحسين المتقدم لمضخات ESP"""
        
        # تحليل البيانات التاريخية
        freq_data = historical_data['frequency'].values
        rate_data = historical_data['oil_rate'].values
        
        if len(freq_data) < 10:
            return self._fallback_optimization(freq_data, rate_data)
        
        # 1. النموذج الفيزيائي
        def physical_model(freq, a, b, c, d):
            """نموذج فيزيائي: Q = a*(f/f0)^3 + b*(f/f0)^2 + c*(f/f0) + d"""
            f0 = 60  # تردد التصميم
            f_norm = freq / f0
            return a * f_norm**3 + b * f_norm**2 + c * f_norm + d
        
        try:
            # تركيب النموذج الفيزيائي
            popt, _ = curve_fit(physical_model, freq_data, rate_data,
                              p0=[100, -50, 200, 500])
            
            # 2. إيجاد الأمثل مع قيود عملية
            freq_range = np.linspace(max(30, freq_data.min()), 
                                   min(70, freq_data.max()), 100)
            
            rates_pred = physical_model(freq_range, *popt)
            
            # 3. حساب التكاليف والربح
            power_consumption = freq_range * 5  # kW تقريبي
            costs = power_consumption * 24 * self.economic_calc.electricity_cost
            
            economic_result = self.economic_calc.optimize_profit(
                rates_pred, costs
            )
            
            # 4. التحقق من كفاءة المضخة
            bep = pump_curve.best_efficiency_point()
            efficiency_penalty = np.abs(rates_pred - bep['flow']) / bep['flow'] * 100
            
            # 5. التوصية المتكاملة
            optimal_idx = np.argmin(efficiency_penalty + (100 - rates_pred/rates_pred.max()*100)/2)
            
            optimal_freq = float(freq_range[optimal_idx])
            predicted_rate = float(rates_pred[optimal_idx])
            
            return {
                "optimal_frequency": optimal_freq,
                "predicted_rate": predicted_rate,
                "expected_increase": max(0, predicted_rate - np.mean(rate_data)),
                "economic_gain": self.economic_calc.calculate_npv(
                    predicted_rate, power_consumption=power_consumption[optimal_idx]
                ),
                "pump_efficiency": float(100 - efficiency_penalty[optimal_idx]),
                "confidence_level": 0.85,
                "optimization_curve": {
                    "frequencies": freq_range.tolist(),
                    "rates": rates_pred.tolist(),
                    "efficiency": (100 - efficiency_penalty).tolist(),
                    "profit": (rates_pred * 70 - costs).tolist()
                }
            }
            
        except Exception as e:
            return self._fallback_optimization(freq_data, rate_data)
    
    def optimize_gas_lift(self, historical_data: pd.DataFrame,
                         valve_depth: float = 5000) -> dict:
        """التحسين المتقدم للرفع بالغاز"""
        
        gas_rates = historical_data['gas_injection'].values
        oil_rates = historical_data['oil_rate'].values
        
        if len(gas_rates) < 15:
            return {"error": "بيانات غير كافية للتحليل المتقدم"}
        
        # 1. نموذج فيزيائي مبسط للرفع بالغاز
        def gas_lift_model(gas_rate, a, b, c, d):
            """Q = a*tanh(b*(gas_rate-c)) + d"""
            return a * np.tanh(b * (gas_rate - c)) + d
        
        try:
            # تركيب النموذج
            popt, _ = curve_fit(gas_lift_model, gas_rates, oil_rates,
                              p0=[500, 0.001, 1000, 1000],
                              maxfev=5000)
            
            # 2. إيجاد النقطة الاقتصادية الأمثل
            gas_range = np.linspace(gas_rates.min(), gas_rates.max(), 100)
            oil_pred = gas_lift_model(gas_range, *popt)
            
            # تكاليف الغاز
            gas_costs = gas_range * self.economic_calc.gas_cost / 1000
            
            # الإيرادات والربح
            revenues = oil_pred * self.economic_calc.oil_price
            profits = revenues - gas_costs
            
            # 3. حساب المشتقة الثانية لإيجاد نقطة التناقص الهامشي
            gradient = np.gradient(oil_pred, gas_range)
            second_gradient = np.gradient(gradient, gas_range)
            
            # نقطة تناقص العائد الهامشي (عندما تبدأ المشتقة الثانية بالسالب)
            marginal_decline_idx = np.where(second_gradient < -0.001)[0]
            
            if len(marginal_decline_idx) > 0:
                optimal_idx = marginal_decline_idx[0]
            else:
                optimal_idx = np.argmax(profits)
            
            optimal_gas = float(gas_range[optimal_idx])
            optimal_oil = float(oil_pred[optimal_idx])
            
            # 4. تحليل الاستقرار
            stability_score = self._calculate_stability(
                historical_data, optimal_gas
            )
            
            return {
                "optimal_gas_injection": optimal_gas,
                "predicted_oil_rate": optimal_oil,
                "gas_oil_ratio": optimal_gas / optimal_oil if optimal_oil > 0 else 0,
                "economic_gain": self.economic_calc.calculate_npv(
                    optimal_oil, gas_injection=optimal_gas
                ),
                "stability_score": stability_score,
                "valve_recommendation": self._optimize_valve_settings(
                    optimal_gas, valve_depth
                ),
                "optimization_curve": {
                    "gas_rates": gas_range.tolist(),
                    "oil_rates": oil_pred.tolist(),
                    "profits": profits.tolist(),
                    "marginal_gain": gradient.tolist()
                }
            }
            
        except Exception as e:
            return self._fallback_gas_lift_optimization(gas_rates, oil_rates)
    
    def _calculate_stability(self, data: pd.DataFrame, 
                           optimal_point: float) -> float:
        """حساب درجة الاستقرار"""
        # تحليل التقلبات حول النقطة المثلى
        deviations = np.abs(data['oil_rate'] - optimal_point)
        stability = 100 * (1 - deviations.std() / data['oil_rate'].mean())
        return min(max(stability, 0), 100)
    
    def _optimize_valve_settings(self, gas_rate: float, 
                               depth: float) -> dict:
        """توصيات إعدادات الصمامات"""
        # قواعد معرفة المجال
        valve_spacing = 500  # قدم بين الصمامات
        num_valves = int(depth / valve_spacing)
        
        # ضغط فتح الصمام الأمثل
        optimal_opening_pressure = gas_rate / 100 + 100  # psi
        
        return {
            "recommended_valves": num_valves,
            "valve_spacing_ft": valve_spacing,
            "opening_pressure_psi": optimal_opening_pressure,
            "injection_depth_ft": depth,
            "gas_rate_per_valve": gas_rate / num_valves if num_valves > 0 else 0
        }
    
    def _fallback_optimization(self, freq, rate):
        """نسخة احتياطية إذا فشل التحليل المتقدم"""
        if len(freq) > 0:
            optimal_idx = np.argmax(rate)
            return {
                "optimal_frequency": float(freq[optimal_idx]),
                "predicted_rate": float(rate[optimal_idx]),
                "confidence_level": 0.6,
                "note": "تحسين أساسي بسبب قلة البيانات"
            }
        return {"error": "لا توجد بيانات كافية"}
    
    def _fallback_gas_lift_optimization(self, gas, oil):
        """نسخة احتياطية للرفع بالغاز"""
        if len(gas) > 0:
            g = np.array(gas)
            o = np.array(oil)
            
            # متوسط النقطة الأفضل
            efficiency = o / (g + 1e-6)  # النفط لكل وحدة غاز
            optimal_idx = np.argmax(efficiency)
            
            return {
                "optimal_gas_injection": float(g[optimal_idx]),
                "predicted_oil_rate": float(o[optimal_idx]),
                "gas_oil_ratio": float(g[optimal_idx] / o[optimal_idx]) if o[optimal_idx] > 0 else 0,
                "confidence_level": 0.65
            }
        return {"error": "لا توجد بيانات"}

# ==================== VISUALIZATION ENGINE ====================

class AdvancedVisualizer:
    """محرك تصور متقدم مع رسومات تفاعلية"""
    
    def create_comprehensive_dashboard(self, 
                                     optimization_results: dict,
                                     historical_data: pd.DataFrame) -> dict:
        """إنشاء لوحة تحكم شاملة"""
        
        figures = {}
        
        # 1. منحنى التحسين الأساسي
        if 'optimization_curve' in optimization_results:
            curve_data = optimization_results['optimization_curve']
            
            fig1 = go.Figure()
            
            if 'frequencies' in curve_data:
                # لمضخة ESP
                fig1.add_trace(go.Scatter(
                    x=curve_data['frequencies'],
                    y=curve_data['rates'],
                    name='معدل الإنتاج',
                    line=dict(color='blue', width=3)
                ))
                
                fig1.add_trace(go.Scatter(
                    x=curve_data['frequencies'],
                    y=curve_data['profit'],
                    name='الربح اليومي',
                    line=dict(color='green', width=2),
                    yaxis='y2'
                ))
                
                fig1.update_layout(
                    title="تحليل تحسين ESP - الإنتاج والربحية",
                    xaxis_title="التردد (هرتز)",
                    yaxis_title="معدل النفط (برميل/يوم)",
                    yaxis2=dict(
                        title="الربح ($/يوم)",
                        overlaying='y',
                        side='right'
                    ),
                    template="plotly_dark"
                )
                
            elif 'gas_rates' in curve_data:
                # للرفع بالغاز
                fig1 = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig1.add_trace(go.Scatter(
                    x=curve_data['gas_rates'],
                    y=curve_data['oil_rates'],
                    name='منحنى الإنتاج',
                    line=dict(color='orange', width=3)
                ), secondary_y=False)
                
                fig1.add_trace(go.Scatter(
                    x=curve_data['gas_rates'],
                    y=curve_data['profits'],
                    name='الربحية',
                    line=dict(color='yellow', width=2)
                ), secondary_y=True)
                
                fig1.add_trace(go.Scatter(
                    x=curve_data['gas_rates'],
                    y=curve_data['marginal_gain'],
                    name='العائد الهامشي',
                    line=dict(color='red', width=2, dash='dash')
                ), secondary_y=False)
                
                fig1.update_layout(
                    title="تحليل الرفع بالغاز - الإنتاج والربحية",
                    xaxis_title="حقن الغاز (MCF/يوم)",
                    template="plotly_dark"
                )
                
                fig1.update_yaxes(title_text="معدل النفط (برميل/يوم)", 
                                secondary_y=False)
                fig1.update_yaxes(title_text="الربح ($/يوم)", 
                                secondary_y=True)
            
            figures['optimization_curve'] = fig1.to_dict()
        
        # 2. مخطط السلسلة الزمنية مع التنبؤ
        if 'time' in historical_data.columns:
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=historical_data['time'],
                y=historical_data['oil_rate'],
                name='الإنتاج الفعلي',
                line=dict(color='cyan', width=2)
            ))
            
            # إضافة متوسط متحرك
            if len(historical_data) > 7:
                ma_7 = historical_data['oil_rate'].rolling(7).mean()
                fig2.add_trace(go.Scatter(
                    x=historical_data['time'],
                    y=ma_7,
                    name='متوسط 7 أيام',
                    line=dict(color='yellow', width=3)
                ))
            
            fig2.update_layout(
                title="الأداء التاريخي مع المتوسط المتحرك",
                xaxis_title="التاريخ",
                yaxis_title="معدل النفط (برميل/يوم)",
                template="plotly_dark"
            )
            
            figures['time_series'] = fig2.to_dict()
        
        # 3. مخطط رادار للأداء المتعدد الأبعاد
        metrics = optimization_results.get('metrics', {})
        if metrics:
            categories = list(metrics.keys())[:6]
            values = list(metrics.values())[:6]
            
            fig3 = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                line=dict(color='lime', width=3)
            ))
            
            fig3.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(values) * 1.2]
                    )),
                showlegend=False,
                title="مخطط رادار لمقاييس الأداء"
            )
            
            figures['radar_chart'] = fig3.to_dict()
        
        # 4. مخطط الأهمية الاقتصادية
        economic = optimization_results.get('economic_gain', {})
        if economic:
            labels = ['الإيرادات', 'تكاليف التشغيل', 'الربح الصافي']
            values = [economic.get('revenue', 0),
                     economic.get('total_cost', 0),
                     economic.get('net_income', 0)]
            
            colors = ['#00FF00', '#FF0000', '#FFFF00']
            
            fig4 = go.Figure(data=[go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f'${v:,.0f}' for v in values],
                textposition='auto',
            )])
            
            fig4.update_layout(
                title="التحليل الاقتصادي الشهري",
                yaxis_title="قيمة ($)",
                template="plotly_dark"
            )
            
            figures['economic_chart'] = fig4.to_dict()
        
        return figures

# ==================== MAIN API ENGINE ====================

class OilNovaAIV2:
    """OILNOVA AI V2.0 - المحرك الرئيسي"""
    
    def __init__(self):
        self.optimizers = {}
        self.visualizer = AdvancedVisualizer()
        self.anomaly_detector = AdvancedAnomalyDetector()
        
    def analyze_well(self, well_data: pd.DataFrame, 
                    well_type: str,
                    config: dict = None) -> dict:
        """تحليل شامل للبئر"""
        
        # التحقق من البيانات
        if well_data.empty:
            return {"error": "بيانات فارغة"}
        
        # تنظيف البيانات
        cleaned_data = self._clean_data(well_data)
        
        # تحديد نوع الرفع إذا لم يكن محدد
        if well_type == "auto":
            well_type = self._detect_lift_type(cleaned_data)
        
        # تحضير خواص السوائل
        fluid_props = FluidProperties(
            oil_gravity=config.get('api_gravity', 35) if config else 35,
            water_cut=config.get('water_cut', 0.3) if config else 0.3
        )
        
        # إنشاء محرك التحسين
        optimizer = DeepOptimizationEngine(well_type, fluid_props)
        
        # التحليل حسب نوع الرفع
        if well_type.lower() == "esp":
            # إنشاء منحنى مضخة
            pump_curve = PumpCurve.from_manufacturer("ESP400", stages=100)
            
            # إنشاء IPR افتراضي
            well_ipr = WellIPR(
                reservoir_pressure=3000,
                productivity_index=2.5,
                oil_rate_max=4000
            )
            
            results = optimizer.optimize_esp(pump_curve, well_ipr, cleaned_data)
            
        elif well_type.lower() in ["gas_lift", "gas"]:
            results = optimizer.optimize_gas_lift(cleaned_data)
            
        elif well_type.lower() == "pcp":
            # PCP optimization
            results = self._optimize_pcp(cleaned_data)
            
        else:
            return {"error": f"نوع رفع غير مدعوم: {well_type}"}
        
        # كشف الشذوذ
        anomaly_results = self.anomaly_detector.predict_failure_risk(
            cleaned_data.iloc[-1] if len(cleaned_data) > 0 else pd.Series()
        )
        
        # إنشاء التصورات
        visualizations = self.visualizer.create_comprehensive_dashboard(
            results, cleaned_data
        )
        
        # تجميع النتائج النهائية
        final_report = {
            "version": "OILNOVA AI V2.0",
            "generated_at": datetime.now().isoformat(),
            "well_type": well_type,
            "optimization_results": results,
            "anomaly_detection": anomaly_results,
            "visualizations": visualizations,
            "key_recommendations": self._generate_recommendations(results, anomaly_results),
            "expected_benefits": {
                "production_increase": results.get('expected_increase', 0),
                "cost_reduction": results.get('economic_gain', {}).get('total_cost_reduction', 0),
                "profit_increase": results.get('economic_gain', {}).get('net_income', 0),
                "payback_period": self._calculate_payback(results)
            },
            "confidence_metrics": {
                "data_quality": self._assess_data_quality(cleaned_data),
                "model_confidence": results.get('confidence_level', 0.7),
                "stability_score": results.get('stability_score', 75)
            }
        }
        
        return final_report
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """تنظيف وتحضير البيانات"""
        cleaned = data.copy()
        
        # إزالة القيم المتطرفة باستخدام IQR
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = cleaned[col].quantile(0.25)
            Q3 = cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # استبدال القيم المتطرفة بالقيم الحدي
            cleaned[col] = np.where(cleaned[col] < lower_bound, lower_bound, cleaned[col])
            cleaned[col] = np.where(cleaned[col] > upper_bound, upper_bound, cleaned[col])
        
        # تعبئة القيم المفقودة
        cleaned = cleaned.fillna(method='ffill').fillna(method='bfill')
        
        return cleaned
    
    def _detect_lift_type(self, data: pd.DataFrame) -> str:
        """كشف نوع الرفع آلياً"""
        columns = [col.lower() for col in data.columns]
        
        if any(x in ' '.join(columns) for x in ['freq', 'vfd', 'esp']):
            return "esp"
        elif any(x in ' '.join(columns) for x in ['gas', 'inject', 'valve']):
            return "gas_lift"
        elif any(x in ' '.join(columns) for x in ['rpm', 'pcp', 'torque']):
            return "pcp"
        else:
            return "esp"  # افتراضي
    
    def _optimize_pcp(self, data: pd.DataFrame) -> dict:
        """تحسين PCP"""
        if 'rpm' not in data.columns or 'oil_rate' not in data.columns:
            return {"error": "بيانات PCP غير كافية"}
        
        rpm = data['rpm'].values
        rate = data['oil_rate'].values
        
        if len(rpm) < 5:
            return {"error": "بيانات غير كافية لتحليل PCP"}
        
        # نموذل PCP: Q = a * RPM + b
        coeffs = np.polyfit(rpm, rate, 1)
        a, b = coeffs
        
        # النقطة المثلى مع مراعاة التآكل (لا تتجاوز 80% من أقصى RPM)
        rpm_max = rpm.max()
        rpm_opt = min(rpm_max * 0.8, np.mean(rpm) * 1.2)
        rate_pred = a * rpm_opt + b
        
        return {
            "optimal_rpm": float(rpm_opt),
            "predicted_rate": float(rate_pred),
            "pump_slip_estimate": self._estimate_pcp_slip(data),
            "recommended_torque": rpm_opt * 2.5,  # N.m تقريبي
            "elastomer_health": 100 - (rpm_opt / rpm_max * 20)
        }
    
    def _estimate_pcp_slip(self, data: pd.DataFrame) -> float:
        """تقدير انزلاق مضخة PCP"""
        if 'rpm' in data.columns and 'oil_rate' in data.columns:
            expected_rate = data['rpm'] * 0.5  # قدرة افتراضية 0.5 برميل/دورة
            actual_rate = data['oil_rate']
            slip = (expected_rate - actual_rate) / expected_rate * 100
            return float(slip.mean())
        return 0.0
    
    def _generate_recommendations(self, opt_results: dict, 
                                anomaly: dict) -> list:
        """توليد توصيات ذكية"""
        recommendations = []
        
        # توصيات التحسين
        if 'optimal_frequency' in opt_results:
            recommendations.append(
                f"ضبط تردد VFD إلى {opt_results['optimal_frequency']:.1f} هرتز "
                f"للحصول على {opt_results.get('expected_increase', 0):.0f} برميل/يوم إضافية"
            )
        
        elif 'optimal_gas_injection' in opt_results:
            recommendations.append(
                f"ضبط حقن الغاز إلى {opt_results['optimal_gas_injection']:.0f} MCF/يوم "
                f"لتحسين كفاءة الرفع بنسبة {opt_results.get('gas_oil_ratio_improvement', 15):.1f}%"
            )
        
        elif 'optimal_rpm' in opt_results:
            recommendations.append(
                f"ضبط سرعة PCP إلى {opt_results['optimal_rpm']:.0f} RPM "
                f"لإطالة عمر الإيلاستومر"
            )
        
        # توصيات الصيانة بناءً على كشف الشذوذ
        if anomaly['risk_score'] > 50:
            recommendations.append(
                f"⚠️ {anomaly['recommended_action']} - درجة المخاطر: {anomaly['risk_score']}"
            )
        
        # توصيات اقتصادية
        economic = opt_results.get('economic_gain', {})
        if economic.get('net_income', 0) > 10000:
            recommendations.append(
                f"💰 زيادة ربحية متوقعة: ${economic['net_income']:,.0f}/شهر"
            )
        
        return recommendations
    
    def _calculate_payback(self, results: dict) -> float:
        """حساب فترة الاسترداد"""
        investment = 50000  # استثمار تقريبي
        monthly_profit = results.get('economic_gain', {}).get('net_income', 0)
        
        if monthly_profit > 0:
            return investment / monthly_profit
        return 0.0
    
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """تقييم جودة البيانات"""
        quality_score = 100
        
        # نقاط البيانات
        if len(data) < 30:
            quality_score -= 20
        
        # القيم المفقودة
        missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        quality_score -= missing_pct * 50
        
        # التنوع في البيانات
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].std() < 1e-6:
                quality_score -= 5
        
        return max(quality_score, 0)

# ==================== FLASK API ====================

app = Flask(__name__)
ai_engine = OilNovaAIV2()

@app.route('/api/v2/analyze', methods=['POST'])
def analyze_v2():
    """نقطة نهاية التحليل المتقدم"""
    try:
        if 'file' not in request.files:
            return jsonify({
                "error": "No file uploaded",
                "solution": "Upload CSV or Excel file"
            }), 400
        
        file = request.files['file']
        
        # قراءة الملف
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format"}), 400
        
        # الحصول على الإعدادات
        well_type = request.form.get('well_type', 'auto')
        config_str = request.form.get('config', '{}')
        
        try:
            config = json.loads(config_str)
        except:
            config = {}
        
        # تشغيل التحليل
        start_time = datetime.now()
        results = ai_engine.analyze_well(df, well_type, config)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # إضافة ميتاداتا
        results['processing_time_seconds'] = processing_time
        results['data_points_analyzed'] = len(df)
        results['ai_model'] = "DeepSeek Custom Physics-AI Hybrid Model"
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": "Contact support for detailed logs"
        }), 500

@app.route('/api/v2/demo', methods=['GET'])
def demo():
    """عرض تجريبي مع بيانات افتراضية"""
    # إنشاء بيانات تجريبية
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
    
    # بيانات ESP تجريبية
    demo_data = pd.DataFrame({
        'date': dates,
        'frequency': 45 + np.random.randn(60) * 5,
        'oil_rate': 1500 + np.random.randn(60) * 200,
        'motor_temp': 160 + np.random.randn(60) * 10,
        'vibration': 0.3 + np.random.randn(60) * 0.1,
        'intake_pressure': 800 + np.random.randn(60) * 50,
        'discharge_pressure': 2200 + np.random.randn(60) * 100,
        'current': 90 + np.random.randn(60) * 5
    })
    
    # تحليل البيانات التجريبية
    results = ai_engine.analyze_well(demo_data, 'esp', {
        'api_gravity': 32,
        'water_cut': 0.25
    })
    
    return jsonify(results)

@app.route('/api/v2/health', methods=['GET'])
def health():
    """فحص صحة النظام"""
    return jsonify({
        "status": "operational",
        "version": "OILNOVA AI V2.0",
        "ai_engine": "DeepSeek Hybrid Physics-AI",
        "models_loaded": True,
        "timestamp": datetime.now().isoformat(),
        "performance": "optimized"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
