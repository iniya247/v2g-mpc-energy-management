import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import milp, LinearConstraint, Bounds
import streamlit as st

st.set_page_config(page_title="Individualized V2G MPC Engineering Workbench", layout="wide")

def run_live_python_mpc(evs, grid_cfg, H_steps=32):
    N = len(evs)
    dt = 0.25
    K = 96
    t_hours = np.arange(K) * dt
    T_amb = 22.0

    P_house = (
        1.2
        + 1.3 * np.exp(-((t_hours - 7.5) ** 2) / (2 * 1.2**2))
        + 3.2 * np.exp(-((t_hours - 19.0) ** 2) / (2 * 1.8**2))
    )
    lambda_buy = np.full(K, 0.22)
    lambda_sell = np.full(K, 0.14)
    off_peak = (t_hours < 6.0) | (t_hours >= 22.0)
    peak = (t_hours >= 17.0) & (t_hours < 22.0)
    lambda_buy[off_peak] = 0.12
    lambda_sell[off_peak] = 0.08
    lambda_buy[peak] = grid_cfg["peak_buy"]
    lambda_sell[peak] = grid_cfg["peak_sell"]

    soc = np.array([e["SOC"] for e in evs], dtype=float)
    soh = np.array([e["SOH"] for e in evs], dtype=float)
    temp = np.array([e["temp_C"] for e in evs], dtype=float)
    budget = np.array([e["deg_budget_kWh"] for e in evs], dtype=float)

    SOC_hist = np.zeros((K + 1, N))
    Temp_hist = np.zeros((K + 1, N))
    Budget_hist = np.zeros((K + 1, N))
    P_chg_hist = np.zeros((K, N))
    P_dis_hist = np.zeros((K, N))
    Action_hist = [[""] * N for _ in range(K)]
    P_grid_hist = np.zeros(K)
    V_pcc_hist = np.zeros(K)

    rev_ev = np.zeros(N)
    chg_cost_ev = np.zeros(N)
    deg_cost_ev = np.zeros(N)
    op_cost_ev = np.zeros(N)
    deg_loss_ev = np.zeros(N)

    SOC_hist[0] = soc
    Temp_hist[0] = temp
    Budget_hist[0] = budget

    alpha_base = 1.5e-4
    gamma_unc = 1.15
    C_rep = 130.0
    C_op = 0.005

    V_nom, V_min, V_max, R_line = 230.0, 218.5, 241.5, 0.35
    P_grid_max_volt = (V_nom - V_min) * V_nom / (R_line * 1000.0)
    P_grid_min_volt = -(V_max - V_nom) * V_nom / (R_line * 1000.0)

    for k in range(K):
        H = min(H_steps, K - k)
        vars_per_block = 4 * N
        total_vars = vars_per_block * H

        c = np.zeros(total_vars)
        lb = np.zeros(total_vars)
        ub = np.zeros(total_vars)
        integrality = np.zeros(total_vars)

        c_deg_nom = np.zeros(N)
        E_eff = np.zeros(N)
        for i in range(N):
            E_eff[i] = evs[i]["E_nom_kWh"] * soh[i]
            f_T = 1.0 + 0.035 * max(0.0, temp[i] - 25.0)
            f_C = 1.0 + 0.80 * (0.5 * evs[i]["P_dis_max_kW"] / evs[i]["E_nom_kWh"])
            f_SOH = 1.0 + 1.50 * (1.0 - soh[i])
            c_deg_nom[i] = alpha_base * f_T * f_C * f_SOH

        for t in range(H):
            k_idx = k + t
            t_hrs = t_hours[k_idx]
            tie = 1.0 + 0.002 * t
            for i in range(N):
                i_pc = t * vars_per_block + i
                i_pd = t * vars_per_block + N + i
                i_uc = t * vars_per_block + 2 * N + i
                i_ud = t * vars_per_block + 3 * N + i
                integrality[i_uc] = 1
                integrality[i_ud] = 1

                avail = t_hrs < evs[i]["departure_hour"]
                if avail:
                    ub[i_pc] = evs[i]["P_chg_max_kW"]
                    ub[i_pd] = evs[i]["P_dis_max_kW"]
                    ub[i_uc] = 1.0
                    ub[i_ud] = 1.0

                deg_rate = C_rep * c_deg_nom[i] + C_op
                term_val = 0.16
                c[i_pc] = (lambda_buy[k_idx] + deg_rate - term_val * evs[i]["eta_chg"]) * dt * tie
                c[i_pd] = (-lambda_sell[k_idx] + deg_rate + term_val / evs[i]["eta_dis"]) * dt / tie

        rows = []
        rhs = []

        for t in range(H):
            for i in range(N):
                i_pc = t * vars_per_block + i
                i_pd = t * vars_per_block + N + i
                i_uc = t * vars_per_block + 2 * N + i
                i_ud = t * vars_per_block + 3 * N + i

                r1 = np.zeros(total_vars); r1[i_uc] = 1; r1[i_ud] = 1
                rows.append(r1); rhs.append(1.0)

                r2 = np.zeros(total_vars); r2[i_pc] = 1; r2[i_uc] = -evs[i]["P_chg_max_kW"]
                rows.append(r2); rhs.append(0.0)

                r3 = np.zeros(total_vars); r3[i_pd] = 1; r3[i_ud] = -evs[i]["P_dis_max_kW"]
                rows.append(r3); rhs.append(0.0)

        for i in range(N):
            dep_step = int(round(evs[i]["departure_hour"] / dt))
            for t in range(H):
                k_idx = k + t
                r_soc = np.zeros(total_vars)
                for tau in range(t + 1):
                    r_soc[tau * vars_per_block + i] = (evs[i]["eta_chg"] * dt) / E_eff[i]
                    r_soc[tau * vars_per_block + N + i] = -(dt / evs[i]["eta_dis"]) / E_eff[i]

                rows.append(r_soc); rhs.append(0.90 - soc[i])

                req_soc = 0.20
                if (k_idx + 1) >= dep_step:
                    req_soc = max(0.20, evs[i]["SOC_dep_min"])
                rows.append(-r_soc); rhs.append(soc[i] - req_soc)

        for i in range(N):
            r_deg = np.zeros(total_vars)
            for t in range(H):
                r_deg[t * vars_per_block + i] = gamma_unc * c_deg_nom[i] * dt
                r_deg[t * vars_per_block + N + i] = gamma_unc * c_deg_nom[i] * dt
            rows.append(r_deg); rhs.append(max(0.0, budget[i]))

        for t in range(H):
            k_idx = k + t
            p_u = min(grid_cfg["P_imp_max"], P_grid_max_volt)
            p_l = max(-grid_cfg["P_exp_max"], P_grid_min_volt)
            r_g = np.zeros(total_vars)
            for i in range(N):
                r_g[t * vars_per_block + i] = 1.0
                r_g[t * vars_per_block + N + i] = -1.0
            rows.append(r_g); rhs.append(p_u - P_house[k_idx])
            rows.append(-r_g); rhs.append(P_house[k_idx] - p_l)

        A = np.array(rows)
        b_u = np.array(rhs)
        b_l = np.full_like(b_u, -np.inf)

        res = milp(c=c, integrality=integrality, bounds=Bounds(lb, ub), constraints=LinearConstraint(A, b_l, b_u))

        P_chg_cmd = np.zeros(N)
        P_dis_cmd = np.zeros(N)
        if res.success and res.x is not None:
            for i in range(N):
                P_chg_cmd[i] = max(0.0, res.x[i])
                P_dis_cmd[i] = max(0.0, res.x[N + i])

        for i in range(N):
            if t_hours[k] >= evs[i]["departure_hour"]:
                P_chg_cmd[i], P_dis_cmd[i] = 0.0, 0.0
                Action_hist[k][i] = "DEPARTED"
            elif P_chg_cmd[i] > 0.05:
                Action_hist[k][i] = "CHARGE"
            elif P_dis_cmd[i] > 0.05:
                Action_hist[k][i] = "DISCHARGE_V2G"
            else:
                Action_hist[k][i] = "IDLE"

            e_eff = evs[i]["E_nom_kWh"] * soh[i]
            soc[i] += (evs[i]["eta_chg"] * P_chg_cmd[i] - P_dis_cmd[i] / evs[i]["eta_dis"]) * dt / e_eff

            p_heat = (1 - evs[i]["eta_chg"]) * P_chg_cmd[i] + (1 / evs[i]["eta_dis"] - 1) * P_dis_cmd[i]
            temp[i] += ((p_heat - (temp[i] - T_amb) / 2.5) / 25.0) * dt

            p_tot = P_chg_cmd[i] + P_dis_cmd[i]
            e_thru = p_tot * dt
            c_rate = p_tot / evs[i]["E_nom_kWh"]
            f_T = 1.0 + 0.035 * max(0.0, temp[i] - 25.0)
            f_C = 1.0 + 0.80 * c_rate
            f_SOH = 1.0 + 1.50 * (1.0 - soh[i])
            dq_nom = alpha_base * f_T * f_C * f_SOH * e_thru

            budget[i] -= gamma_unc * dq_nom
            soh[i] -= dq_nom / evs[i]["E_nom_kWh"]

            SOC_hist[k + 1, i] = soc[i]
            Temp_hist[k + 1, i] = temp[i]
            Budget_hist[k + 1, i] = budget[i]
            P_chg_hist[k, i] = P_chg_cmd[i]
            P_dis_hist[k, i] = P_dis_cmd[i]

            rev_ev[i] += (P_dis_cmd[i] * dt) * lambda_sell[k]
            chg_cost_ev[i] += (P_chg_cmd[i] * dt) * lambda_buy[k]
            deg_cost_ev[i] += dq_nom * C_rep
            op_cost_ev[i] += e_thru * C_op
            deg_loss_ev[i] += dq_nom

        P_grid_hist[k] = P_house[k] + np.sum(P_chg_cmd) - np.sum(P_dis_cmd)
        V_pcc_hist[k] = V_nom - (R_line * P_grid_hist[k] * 1000.0) / V_nom

    return {
        "t_hours": t_hours,
        "P_house": P_house,
        "lambda_buy": lambda_buy,
        "lambda_sell": lambda_sell,
        "SOC_hist": SOC_hist,
        "Temp_hist": Temp_hist,
        "Budget_hist": Budget_hist,
        "P_chg_hist": P_chg_hist,
        "P_dis_hist": P_dis_hist,
        "Action_hist": Action_hist,
        "P_grid_hist": P_grid_hist,
        "V_pcc_hist": V_pcc_hist,
        "rev_ev": rev_ev,
        "chg_cost_ev": chg_cost_ev,
        "deg_cost_ev": deg_cost_ev,
        "op_cost_ev": op_cost_ev,
        "net_profit_ev": rev_ev - chg_cost_ev - deg_cost_ev - op_cost_ev,
        "deg_loss_ev": deg_loss_ev,
    }


st.title("Individualized Battery Degradation-Budget-Constrained V2G MPC")
st.caption("Residential EV Fleet Energy Management System | MILP Optimization + Protocol Telemetry")

with st.sidebar:
    st.header("1. Preset Research Scenarios")
    preset = st.selectbox(
        "Load Scenario Preset",
        [
            "Scenario A: Average-Health EV Selected",
            "Scenario B: Healthy EV Selected",
            "Scenario C: No EV Selected (Constrained)",
            "Scenario D: Multi-EV + Voltage Clamping",
        ],
    )

    if "Scenario A" in preset:
        defs = [(0.55, 0.96, 24.0, 7.0, 0.85, 0.015), (0.85, 0.86, 26.5, 22.0, 0.45, 0.025), (0.65, 0.81, 29.0, 21.0, 0.60, 0.001)]
    elif "Scenario B" in preset:
        defs = [(0.88, 0.96, 24.0, 23.0, 0.50, 0.030), (0.75, 0.86, 26.5, 18.5, 0.70, 0.001), (0.65, 0.81, 29.0, 20.0, 0.60, 0.001)]
    elif "Scenario C" in preset:
        defs = [(0.75, 0.96, 24.0, 19.0, 0.75, 0.0002), (0.60, 0.86, 26.5, 19.5, 0.60, 0.0002), (0.65, 0.81, 29.0, 20.0, 0.65, 0.0002)]
    else:
        defs = [(0.88, 0.96, 24.0, 23.0, 0.50, 0.030), (0.88, 0.86, 26.5, 23.0, 0.45, 0.030), (0.85, 0.81, 29.0, 23.0, 0.45, 0.025)]

    st.header("2. Individual EV Parameters")
    ev_configs = []
    names = ["EV_1 (60 kWh Healthy)", "EV_2 (50 kWh Avg-Health)", "EV_3 (40 kWh Aged)"]
    caps = [60.0, 50.0, 40.0]
    pmax = [7.4, 7.4, 6.6]
    etas = [0.94, 0.92, 0.90]

    for idx in range(3):
        with st.expander(names[idx], expanded=(idx == 1)):
            s_soc = st.slider(f"Initial SOC (%) - EV_{idx+1}", 25, 90, int(defs[idx][0] * 100)) / 100.0
            s_soh = st.slider(f"SOH (%) - EV_{idx+1}", 75, 100, int(defs[idx][1] * 100)) / 100.0
            s_tmp = st.slider(f"Temp (°C) - EV_{idx+1}", 15.0, 40.0, float(defs[idx][2]))
            s_dep = st.slider(f"Departure Hour - EV_{idx+1}", 6.0, 23.5, float(defs[idx][3]), step=0.5)
            s_dmin = st.slider(f"Min Departure SOC (%) - EV_{idx+1}", 30, 90, int(defs[idx][4] * 100)) / 100.0
            s_bdg = st.number_input(f"Deg. Budget (kWh) - EV_{idx+1}", 0.0000, 0.1000, float(defs[idx][5]), format="%.4f", step=0.001)
            ev_configs.append({
                "id": f"EV_{idx+1}",
                "E_nom_kWh": caps[idx],
                "SOH": s_soh,
                "SOC": s_soc,
                "temp_C": s_tmp,
                "P_chg_max_kW": pmax[idx],
                "P_dis_max_kW": pmax[idx],
                "eta_chg": etas[idx],
                "eta_dis": etas[idx],
                "departure_hour": s_dep,
                "SOC_dep_min": s_dmin,
                "deg_budget_kWh": s_bdg,
            })

    st.header("3. DSO Grid & Tariff Controls")
    grid_cfg = {
        "P_imp_max": st.slider("DSO Max Grid Import (kW)", 8.0, 25.0, 15.0),
        "P_exp_max": st.slider("DSO Max Grid Export (kW)", 4.0, 20.0, 12.0),
        "peak_buy": st.slider("Peak Import Tariff ($/kWh)", 0.25, 0.70, 0.45),
        "peak_sell": st.slider("Peak V2G Export Tariff ($/kWh)", 0.15, 0.65, 0.38),
    }

sim = run_live_python_mpc(ev_configs, grid_cfg)

tot_rev = np.sum(sim["rev_ev"])
tot_chg = np.sum(sim["chg_cost_ev"])
tot_deg = np.sum(sim["deg_cost_ev"])
tot_net = np.sum(sim["net_profit_ev"])
max_v = np.max(sim["V_pcc_hist"])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Fleet Net Profit (24h)", f"${tot_net:.2f}")
c2.metric("V2G Export Revenue", f"+${tot_rev:.2f}")
c3.metric("Charging Expenditure", f"-${tot_chg:.2f}")
c4.metric("Battery Degradation Cost", f"-${tot_deg:.2f}")
c5.metric("Max PCC Feeder Voltage", f"{max_v:.2f} V", delta="Within 241.5V Limit" if max_v <= 241.55 else "Overvoltage!")

st.subheader("Individual EV Fleet Status & Economics Breakdown")
ev_rows = []
for i, e in enumerate(ev_configs):
    v2g_kwh = np.sum(sim["P_dis_hist"][:, i]) * 0.25
    chg_kwh = np.sum(sim["P_chg_hist"][:, i]) * 0.25
    rem_bdg = sim["Budget_hist"][-1, i]
    ev_rows.append({
        "Vehicle": e["id"],
        "SOH (%)": f"{e['SOH']*100:.1f}%",
        "Usable Cap (kWh)": f"{e['E_nom_kWh']*e['SOH']:.2f}",
        "Init -> Final SOC": f"{e['SOC']*100:.0f}% -> {sim['SOC_hist'][-1, i]*100:.1f}%",
        "Departure": f"{e['departure_hour']:.1f}h (Min {e['SOC_dep_min']*100:.0f}%)",
        "V2G Export (kWh)": round(v2g_kwh, 2),
        "Charged (kWh)": round(chg_kwh, 2),
        "Rem. Budget (Wh)": round(rem_bdg * 1000, 3),
        "Export Rev ($)": round(sim["rev_ev"][i], 2),
        "Deg. Cost ($)": round(sim["deg_cost_ev"][i], 2),
        "Net Profit ($)": round(sim["net_profit_ev"][i], 2),
    })
st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Individual EV SOC Trajectories (%)",
        "Individual EV Dispatch Power (+Charge / -V2G Export kW)",
        "Remaining Degradation Budget (Wh)",
        "Residential PCC Voltage (V) & Net Grid Power (kW)",
    ),
)

colors = ["#1f77b4", "#2ca02c", "#d62728"]
for i in range(3):
    fig.add_trace(go.Scatter(x=sim["t_hours"], y=sim["SOC_hist"][:-1, i] * 100, name=f"EV_{i+1} SOC", line=dict(color=colors[i], width=2)), row=1, col=1)
    net_p = sim["P_chg_hist"][:, i] - sim["P_dis_hist"][:, i]
    fig.add_trace(go.Scatter(x=sim["t_hours"], y=net_p, name=f"EV_{i+1} Power", line=dict(color=colors[i], width=2), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=sim["t_hours"], y=sim["Budget_hist"][:-1, i] * 1000, name=f"EV_{i+1} Budget", line=dict(color=colors[i], width=2), showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=sim["t_hours"], y=sim["V_pcc_hist"], name="PCC Voltage (V)", line=dict(color="#9467bd", width=2.5)), row=2, col=2)
fig.add_hline(y=241.5, line_dash="dash", line_color="red", annotation_text="Max 241.5V", row=2, col=2)

fig.update_layout(height=650, template="plotly_white", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("Live Protocol Telemetry Log (CAN / ISO 15118 / OCPP / APIs)")
    inspect_hr = st.slider("Inspect Telemetry Timestamp (Hour of Day)", 0.0, 23.75, 18.0, step=0.25)
    k_sel = int(round(inspect_hr / 0.25))
    ts = f"2026-11-15T{int(inspect_hr):02d}:{int((inspect_hr%1)*60):02d}:00Z"

    logs = [
        f"[{ts}] [CAN 0x2F1 | BMS->Gateway] EV_2: Temp={sim['Temp_hist'][k_sel,1]:.1f}C, SOH={ev_configs[1]['SOH']*100:.1f}%, SOC={sim['SOC_hist'][k_sel,1]*100:.1f}%",
        f"[{ts}] [ISO 15118-20 | EV_2->EVSE_2] Mode=BPT_AC, Action={sim['Action_hist'][k_sel][1]}, Dep={ev_configs[1]['departure_hour']:.1f}h",
        f"[{ts}] [OCPP 2.0.1 | EMS->EVSE_2] SetChargingProfile: P_chg={sim['P_chg_hist'][k_sel,1]:.2f}kW, P_dis={sim['P_dis_hist'][k_sel,1]:.2f}kW",
        f"[{ts}] [DSO Utility API | Grid->EMS] P_grid={sim['P_grid_hist'][k_sel]:.2f}kW, V_PCC={sim['V_pcc_hist'][k_sel]:.2f}V, Sell=${sim['lambda_sell'][k_sel]:.2f}/kWh",
        f"[{ts}] [OEM Cloud API | Authorized Scope] EV_1: Plugged={inspect_hr < ev_configs[0]['departure_hour']}, SOC={sim['SOC_hist'][k_sel,0]*100:.1f}% (Raw cell V restricted)",
    ]
    st.code("\n".join(logs), language="text")

with col_right:
    st.subheader("MATLAB Benchmark Validation (Exported Dataset)")
    if os.path.exists("v2g_sim_export.json"):
        with open("v2g_sim_export.json", "r") as f:
            mdata = json.load(f)
        bench_df = pd.DataFrame([
            {"Method": "Proposed Budget-MPC (MATLAB)", "Net Profit ($)": round(mdata["ScenarioA"]["total_profit"], 2), "Deg. Cost ($)": round(mdata["ScenarioA"]["total_deg_cost"], 2), "Grid Violations": mdata["ScenarioA"]["grid_violations"]},
            {"Method": "SOH-MPC No Budget (MATLAB)", "Net Profit ($)": round(mdata["ScenarioA_NoBudget"]["total_profit"], 2), "Deg. Cost ($)": round(mdata["ScenarioA_NoBudget"]["total_deg_cost"], 2), "Grid Violations": mdata["ScenarioA_NoBudget"]["grid_violations"]},
            {"Method": "Conventional SOC Rule (MATLAB)", "Net Profit ($)": round(mdata["ScenarioA_Conv"]["total_profit"], 2), "Deg. Cost ($)": round(mdata["ScenarioA_Conv"]["total_deg_cost"], 2), "Grid Violations": mdata["ScenarioA_Conv"]["grid_violations"]},
        ])
        st.dataframe(bench_df, use_container_width=True, hide_index=True)
        st.success("Loaded verified MATLAB R2026b export (`v2g_sim_export.json`).")
    else:
        st.info("Place `v2g_sim_export.json` from MATLAB Drive in this folder to view side-by-side MATLAB benchmark metrics.")