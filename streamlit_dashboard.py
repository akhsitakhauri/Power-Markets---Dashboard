import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta


st.set_page_config(page_title="European Power Market", layout="wide")

st.title("European Power Market — Supply, Demand & Pricing Model")

st.markdown(
    """
    Interactive model to explore generation mix, demand and marginal costs
    determine hourly market clearing prices in a simplified European market.
    The app uses a synthetic demand and availability profile so it runs without external data.
    """
)


def generate_time_index(start_date, end_date, freq='H'):
    return pd.date_range(start=start_date, end=end_date, freq=freq)


DEFAULT_COUNTRIES = ['Germany', 'France', 'Spain', 'Italy', 'Netherlands']

base_demand_mw = {
    'Germany': 65000,
    'France': 48000,
    'Spain': 28000,
    'Italy': 30000,
    'Netherlands': 15000,
}

# Rough installed capacities by technology per country (MW) — synthetic defaults
default_capacity = {
    'wind': {'Germany': 60000, 'France': 18000, 'Spain': 25000, 'Italy': 11000, 'Netherlands': 8000},
    'solar': {'Germany': 54000, 'France': 12000, 'Spain': 30000, 'Italy': 22000, 'Netherlands': 4000},
    'hydro': {'Germany': 6000, 'France': 25000, 'Spain': 20000, 'Italy': 18000, 'Netherlands': 1500},
    'thermal': {'Germany': 80000, 'France': 60000, 'Spain': 40000, 'Italy': 42000, 'Netherlands': 24000},
}


def availability_profiles(index):
    # Diurnal and seasonal pattern for solar, wind random-ish, hydro seasonal, thermal firm
    hours = index.hour.values
    day_of_year = index.dayofyear.values
    # Solar: +ve during daytime, scaled by day-of-year
    solar = np.clip(np.sin((hours - 6) / 24 * 2 * np.pi), 0, None)
    solar *= (0.5 + 0.5 * np.cos((day_of_year - 172) / 365 * 2 * np.pi))  # more in summer
    # Wind: random with slight seasonal variation
    rng = np.random.default_rng(42)
    wind = 0.35 + 0.15 * np.sin((day_of_year / 365) * 2 * np.pi) + rng.normal(0, 0.1, len(index))
    wind = np.clip(wind, 0.05, 0.95)
    # Hydro: seasonal
    hydro = 0.4 + 0.2 * np.cos((day_of_year - 80) / 365 * 2 * np.pi)
    hydro = np.clip(hydro, 0.05, 0.95)
    # Thermal: mostly always available but with maintenance dips
    thermal = 0.9 + 0.03 * np.sin((hours) / 24 * 2 * np.pi)
    thermal = np.clip(thermal, 0.4, 1.0)
    return {'solar': solar, 'wind': wind, 'hydro': hydro, 'thermal': thermal}


def build_market(index, countries, tech_caps, tech_marginal_costs, demand_scale=1.0, elasticity=0.0):
    profiles = availability_profiles(index)
    rows = []
    for t in index:
        hour_idx = index.get_loc(t)
        total_demand = 0.0
        supply_by_tech = {tech: 0.0 for tech in tech_caps}
        # aggregate across countries
        for c in countries:
            country_base = base_demand_mw.get(c, 10000)
            # simple diurnal demand shape: higher during day/evening
            h = t.hour
            diurnal = 1.0 + 0.12 * np.sin((h - 15) / 24 * 2 * np.pi) + 0.05 * np.cos((h - 5) / 24 * 2 * np.pi)
            seasonal = 1.0 + 0.08 * np.cos((t.timetuple().tm_yday - 30) / 365 * 2 * np.pi)
            demand = country_base * diurnal * seasonal * demand_scale
            total_demand += demand
            # generation available
            for tech, caps in tech_caps.items():
                cap = caps.get(c, 0)
                avail = profiles[tech][hour_idx]
                supply_by_tech[tech] += cap * avail

        # Build supply stack and compute clearing price
        # Represent supply as small blocks per technology (aggregate continuous approximation)
        tech_items = []
        for tech, supplied in supply_by_tech.items():
            tech_items.append({'tech': tech, 'supply': supplied, 'mc': tech_marginal_costs[tech]})
        # sort by marginal cost ascending
        tech_items = sorted(tech_items, key=lambda x: x['mc'])
        cum = 0.0
        price = tech_items[-1]['mc']  # default price if demand > supply
        dispatched = {tech: 0.0 for tech in supply_by_tech}
        for item in tech_items:
            prev_cum = cum
            cum += item['supply']
            if cum >= total_demand:
                # marginal unit in this technology
                price = item['mc']
                needed = max(0.0, total_demand - prev_cum)
                dispatched[item['tech']] += needed
                break
            else:
                dispatched[item['tech']] += item['supply']

        # if supply > demand, the remaining cheapest units are curtailed (no negative prices here)
        rows.append({'timestamp': t, 'demand_mw': total_demand, 'price_eur_per_mwh': price, **{f'{k}_gen_mw': v for k, v in dispatched.items()}})

    df = pd.DataFrame(rows).set_index('timestamp')
    # add total generation column
    gen_cols = [c for c in df.columns if c.endswith('_gen_mw')]
    df['total_generation_mw'] = df[gen_cols].sum(axis=1)
    return df


with st.sidebar:
    st.header("Model inputs")
    countries = st.multiselect("Select countries", options=DEFAULT_COUNTRIES, default=DEFAULT_COUNTRIES)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=(datetime.today() - timedelta(days=7)).date())
        end_date = st.date_input("End date", value=datetime.today().date())
    with col2:
        freq = st.selectbox("Time resolution", ['H', 'D'], index=0)
        demand_scale = st.slider("Demand scale (relative)", 0.5, 1.5, 1.0, 0.01)

    st.subheader("Technology settings")
    mc_wind = st.number_input("Wind marginal cost (€/MWh)", value=5.0)
    mc_solar = st.number_input("Solar marginal cost (€/MWh)", value=5.0)
    mc_hydro = st.number_input("Hydro marginal cost (€/MWh)", value=20.0)
    mc_thermal = st.number_input("Thermal marginal cost (€/MWh)", value=60.0)

    st.markdown("**Installed capacity scale (fraction of defaults)**")
    cap_scale = st.slider("Capacity scale", 0.2, 2.0, 1.0)

    if not countries:
        st.warning("Select at least one country to run the model.")

run_button = st.sidebar.button("Run model")

if run_button and countries:
    # Build time index
    idx = generate_time_index(start_date, end_date + timedelta(days=1) if freq == 'D' else end_date + timedelta(hours=23), freq=freq)
    # scale capacities
    tech_caps = {}
    for tech, table in default_capacity.items():
        tech_caps[tech] = {c: int(table.get(c, 0) * cap_scale) for c in countries}

    tech_mcs = {'wind': mc_wind, 'solar': mc_solar, 'hydro': mc_hydro, 'thermal': mc_thermal}

    with st.spinner('Simulating market...'):
        df = build_market(idx, countries, tech_caps, tech_mcs, demand_scale=demand_scale)

    st.success('Simulation complete')

    # Plots
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(df.reset_index(), x='timestamp', y='demand_mw', title='Total Demand (MW)')
        st.plotly_chart(fig1, use_container_width=True)

        # stacked generation
        gen_cols = [c for c in df.columns if c.endswith('_gen_mw')]
        gen_df = df[gen_cols].rename(columns=lambda x: x.replace('_gen_mw', ''))
        gen_df = gen_df.reset_index().melt(id_vars='timestamp', var_name='tech', value_name='mw')
        fig_area = px.area(gen_df, x='timestamp', y='mw', color='tech', title='Dispatched Generation by Tech (MW)')
        st.plotly_chart(fig_area, use_container_width=True)

    with col2:
        fig2 = px.line(df.reset_index(), x='timestamp', y='price_eur_per_mwh', title='Market Clearing Price (€/MWh)')
        st.plotly_chart(fig2, use_container_width=True)

        # summary statistics
        st.metric('Average Price (€/MWh)', f"{df['price_eur_per_mwh'].mean():.1f}")
        st.metric('Average Demand (MW)', f"{df['demand_mw'].mean():.0f}")

    st.subheader('Data & Export')
    st.dataframe(df.head(200))
    csv = df.reset_index().to_csv(index=False)
    st.download_button('Download simulation CSV', csv, file_name='simulation.csv', mime='text/csv')

    st.markdown('---')
    st.markdown('Notes: This is a simplified model (no transmission, no unit commitment, no negative prices).')

    # mark todo item completed in-app (visual only)
    st.info('Model run completed — see README for details on extending the model.')

else:
    st.info('Configure inputs in the sidebar and click "Run model".')
