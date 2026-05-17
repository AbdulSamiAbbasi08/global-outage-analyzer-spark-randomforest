import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import joblib
import numpy as np
import os
import sys
import io

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("src")

from forecast import get_7day_forecast

print("Loading data and model...")

df = pd.read_csv("data/cleaned_outage_data.csv")
model = joblib.load("models/rf_model.pkl")
metrics = pd.read_csv("models/metrics.csv")
importance = pd.read_csv("models/feature_importance.csv")

countries = sorted(df["country"].unique().tolist())

app = dash.Dash(__name__)
app.title = "Global Internet Outage Analyzer"

DAY_NAMES = ["Today", "Tomorrow", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]

app.layout = html.Div(style={"fontFamily": "Arial", "backgroundColor": "#0f1117", "color": "white", "minHeight": "100vh", "padding": "20px"}, children=[

    # Header
    html.Div([
        html.H1("🌐 Global Internet Outage Analyzer", style={"textAlign": "center", "color": "#00d4ff", "marginBottom": "5px"}),
        html.P("Powered by Apache Spark + Random Forest | Big Data Analytics Project", style={"textAlign": "center", "color": "#888", "marginTop": "0"}),
    ]),

    # Metrics bar
    html.Div([
        html.Div([html.H3(f"{metrics['accuracy'][0]*100:.1f}%", style={"color": "#00ff88", "margin": "0"}), html.P("Accuracy", style={"margin": "0", "color": "#888"})],
                 style={"textAlign": "center", "padding": "15px", "backgroundColor": "#1a1a2e", "borderRadius": "10px", "flex": "1"}),
        html.Div([html.H3(f"{metrics['precision'][0]*100:.1f}%", style={"color": "#00d4ff", "margin": "0"}), html.P("Precision", style={"margin": "0", "color": "#888"})],
                 style={"textAlign": "center", "padding": "15px", "backgroundColor": "#1a1a2e", "borderRadius": "10px", "flex": "1"}),
        html.Div([html.H3(f"{metrics['recall'][0]*100:.1f}%", style={"color": "#ffaa00", "margin": "0"}), html.P("Recall", style={"margin": "0", "color": "#888"})],
                 style={"textAlign": "center", "padding": "15px", "backgroundColor": "#1a1a2e", "borderRadius": "10px", "flex": "1"}),
        html.Div([html.H3(f"{metrics['f1_score'][0]*100:.1f}%", style={"color": "#ff6b6b", "margin": "0"}), html.P("F1 Score", style={"margin": "0", "color": "#888"})],
                 style={"textAlign": "center", "padding": "15px", "backgroundColor": "#1a1a2e", "borderRadius": "10px", "flex": "1"}),
        html.Div([html.H3("60,000", style={"color": "#bb86fc", "margin": "0"}), html.P("Records", style={"margin": "0", "color": "#888"})],
                 style={"textAlign": "center", "padding": "15px", "backgroundColor": "#1a1a2e", "borderRadius": "10px", "flex": "1"}),
    ], style={"display": "flex", "gap": "15px", "marginBottom": "20px"}),

    # World map
    html.Div([
        html.H3("🗺️ Global Outage Risk Map", style={"color": "#00d4ff"}),
        dcc.Graph(id="world-map"),
    ], style={"backgroundColor": "#1a1a2e", "padding": "20px", "borderRadius": "10px", "marginBottom": "20px"}),

    # Country selector
    html.Div([
        html.H3("🔍 Select Country for Forecast", style={"color": "#00d4ff", "margin": "0"}),
        html.P("Weather is fetched automatically — no manual input needed", style={"color": "#888", "margin": "5px 0 15px 0"}),
        dcc.Dropdown(
            id="country-dropdown",
            options=[{"label": c, "value": c} for c in countries],
            value="Pakistan",
            style={"backgroundColor": "#0f1117", "color": "black", "maxWidth": "400px"}
        ),
    ], style={"backgroundColor": "#1a1a2e", "padding": "20px", "borderRadius": "10px", "marginBottom": "20px"}),

    # 7-day forecast cards
    html.Div([
        html.H3("📅 7-Day Internet Outage Forecast", style={"color": "#00d4ff"}),
        html.Div([
            html.Button(
                "📊 Next 7 Days Prediction",
                id="btn-7day",
                n_clicks=0,
                style={
                    "backgroundColor": "#00d4ff",
                    "color": "#0f1117",
                    "border": "none",
                    "borderRadius": "8px",
                    "padding": "10px 20px",
                    "fontWeight": "bold",
                    "fontSize": "14px",
                    "cursor": "pointer",
                    "marginBottom": "15px",
                }
            ),
        ]),
        html.P("Click a day card to see hourly breakdown", style={"color": "#888", "marginTop": "0"}),
        dcc.Loading(
            id="forecast-loading",
            type="circle",
            color="#00d4ff",
            children=html.Div(id="forecast-cards", style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
        ),
    ], style={"backgroundColor": "#1a1a2e", "padding": "20px", "borderRadius": "10px", "marginBottom": "20px"}),

    # Hourly / overview chart
    html.Div([
        html.H3(id="hourly-title", children="⏰ Click a day card or the 7-Day button above", style={"color": "#00d4ff"}),
        dcc.Graph(id="hourly-chart"),
    ], style={"backgroundColor": "#1a1a2e", "padding": "20px", "borderRadius": "10px", "marginBottom": "20px"}),

    # History + top countries
    html.Div([
        html.Div([
            html.H3("📈 Outage History", style={"color": "#00d4ff"}),
            dcc.Graph(id="country-history"),
        ], style={"flex": "2", "backgroundColor": "#1a1a2e", "padding": "20px", "borderRadius": "10px"}),

        html.Div([
            html.H3("🏆 Top 15 by Outage Rate", style={"color": "#00d4ff"}),
            dcc.Graph(id="top-countries"),
        ], style={"flex": "1", "backgroundColor": "#1a1a2e", "padding": "20px", "borderRadius": "10px"}),
    ], style={"display": "flex", "gap": "15px", "marginBottom": "20px"}),

    # Feature importance
    html.Div([
        html.H3("🧠 Feature Importance", style={"color": "#00d4ff"}),
        dcc.Graph(id="feature-importance"),
    ], style={"backgroundColor": "#1a1a2e", "padding": "20px", "borderRadius": "10px"}),

    # Hidden data store
    dcc.Store(id="forecast-store"),
])


# ── Callbacks ──

@app.callback(Output("world-map", "figure"), Input("country-dropdown", "value"))
def update_map(_):
    try:
        iso2_to_iso3 = {
            "AF": "AFG", "AU": "AUS", "BD": "BGD", "BR": "BRA",
            "CA": "CAN", "CN": "CHN", "CU": "CUB", "DE": "DEU",
            "EG": "EGY", "ES": "ESP", "ET": "ETH", "FR": "FRA",
            "GB": "GBR", "ID": "IDN", "IN": "IND", "IR": "IRN",
            "IT": "ITA", "JP": "JPN", "KR": "KOR", "MM": "MMR",
            "MX": "MEX", "NG": "NGA", "PH": "PHL", "PK": "PAK",
            "RU": "RUS", "SA": "SAU", "TR": "TUR", "UA": "UKR",
            "US": "USA", "VE": "VEN",
        }
        country_stats = df.groupby(["country", "iso_code"])["outage"].mean().reset_index()
        country_stats.columns = ["country", "iso_code", "outage_rate"]
        country_stats["outage_pct"] = (country_stats["outage_rate"] * 100).round(1)
        country_stats["iso3"] = country_stats["iso_code"].map(iso2_to_iso3)

        fig = px.choropleth(
            country_stats, locations="iso3", color="outage_pct",
            hover_name="country", locationmode="ISO-3",
            color_continuous_scale=["#00ff88", "#ffaa00", "#ff4444"],
            range_color=[0, 80], labels={"outage_pct": "Risk %"},
        )
        fig.update_geos(
            showframe=False, showcoastlines=True, coastlinecolor="#888",
            showland=True, landcolor="#3a3a3a",
            showocean=True, oceancolor="#1a1a2e",
            projection_type="equirectangular",
        )
        fig.update_layout(
            paper_bgcolor="#1a1a2e", font_color="white",
            margin=dict(l=0, r=0, t=0, b=0), height=420,
            coloraxis_colorbar=dict(
                title=dict(text="Risk %", font=dict(color="white")),
                tickfont=dict(color="white"),
            ),
        )
        return fig
    except Exception as e:
        print(f"Map error: {e}")
        return go.Figure()


@app.callback(
    Output("forecast-store", "data"),
    Output("forecast-cards", "children"),
    Input("country-dropdown", "value")
)
def update_forecast(country):
    try:
        forecast_df = get_7day_forecast(country, model, df)
        if forecast_df is None:
            return None, [html.P("No data for this country", style={"color": "red"})]

        daily = forecast_df.groupby("day")["risk"].mean().reset_index()

        cards = []
        for _, row in daily.iterrows():
            day_num = int(row["day"])
            avg_risk = row["risk"]

            if avg_risk < 30:
                color = "#00ff88"
                emoji = "🟢"
                label = "LOW"
            elif avg_risk < 60:
                color = "#ffaa00"
                emoji = "🟡"
                label = "MED"
            else:
                color = "#ff4444"
                emoji = "🔴"
                label = "HIGH"

            card = html.Div(
                id={"type": "day-card", "index": day_num},
                children=[
                    html.P(DAY_NAMES[day_num], style={"margin": "0", "color": "#888", "fontSize": "12px"}),
                    html.H2(emoji, style={"margin": "5px 0", "fontSize": "30px"}),
                    html.H3(f"{avg_risk:.0f}%", style={"margin": "0", "color": color}),
                    html.P(label, style={"margin": "0", "color": color, "fontSize": "12px"}),
                ],
                n_clicks=0,
                style={
                    "backgroundColor": "#0f1117",
                    "border": f"2px solid {color}",
                    "borderRadius": "10px",
                    "padding": "15px",
                    "textAlign": "center",
                    "cursor": "pointer",
                    "flex": "1",
                    "minWidth": "100px",
                }
            )
            cards.append(card)

        return forecast_df.to_json(), cards

    except Exception as e:
        print(f"Forecast error: {e}")
        return None, [html.P(f"Forecast error: {e}", style={"color": "red"})]


@app.callback(
    Output("hourly-chart", "figure"),
    Output("hourly-title", "children"),
    Input("btn-7day", "n_clicks"),
    Input({"type": "day-card", "index": dash.ALL}, "n_clicks"),
    State("forecast-store", "data"),
    State("country-dropdown", "value"),
    prevent_initial_call=True
)
def update_hourly(btn_clicks, n_clicks_list, forecast_json, country):
    try:
        if forecast_json is None:
            return go.Figure(), "⏰ Click a day card or the 7-Day button above"

        forecast_df = pd.read_json(io.StringIO(forecast_json))
        triggered = ctx.triggered_id

        # ── 7-day overview bar chart ──
        if triggered == "btn-7day":
            daily = forecast_df.groupby("day")["risk"].mean().reset_index()
            daily["day_name"] = daily["day"].apply(lambda d: DAY_NAMES[d])

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=daily["day_name"],
                y=daily["risk"],
                marker_color=daily["risk"].apply(
                    lambda r: "#ff4444" if r >= 60 else "#ffaa00" if r >= 30 else "#00ff88"
                ),
                hovertemplate="%{x}<br>Avg Risk: %{y:.1f}%<extra></extra>"
            ))

            fig.add_hrect(y0=0,  y1=30,  fillcolor="#00ff88", opacity=0.05, line_width=0)
            fig.add_hrect(y0=30, y1=60,  fillcolor="#ffaa00", opacity=0.05, line_width=0)
            fig.add_hrect(y0=60, y1=100, fillcolor="#ff4444", opacity=0.05, line_width=0)

            fig.update_layout(
                paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                font_color="white", height=300,
                xaxis=dict(gridcolor="#333"),
                yaxis=dict(title="Avg Outage Risk %", gridcolor="#333", range=[0, 100]),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            return fig, f"📊 7-Day Overview — {country}"

        # ── Individual day hourly line chart ──
        if not isinstance(triggered, dict) or triggered.get("type") != "day-card":
            return go.Figure(), "⏰ Click a day card or the 7-Day button above"

        day_num = triggered["index"]
        day_data = forecast_df[forecast_df["day"] == day_num]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=day_data["hour"],
            y=day_data["risk"],
            mode="lines+markers",
            line=dict(color="#00d4ff", width=2),
            marker=dict(
                color=day_data["risk"].apply(
                    lambda r: "#ff4444" if r >= 60 else "#ffaa00" if r >= 30 else "#00ff88"
                ),
                size=8
            ),
            hovertemplate="Hour: %{x}:00<br>Risk: %{y}%<extra></extra>"
        ))

        fig.add_hrect(y0=0,  y1=30,  fillcolor="#00ff88", opacity=0.05, line_width=0)
        fig.add_hrect(y0=30, y1=60,  fillcolor="#ffaa00", opacity=0.05, line_width=0)
        fig.add_hrect(y0=60, y1=100, fillcolor="#ff4444", opacity=0.05, line_width=0)

        fig.update_layout(
            paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
            font_color="white", height=300,
            xaxis=dict(title="Hour of Day", gridcolor="#333", tickmode="linear", tick0=0, dtick=3),
            yaxis=dict(title="Outage Risk %", gridcolor="#333", range=[0, 100]),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        return fig, f"⏰ Hourly Risk — {country} — {DAY_NAMES[day_num]}"

    except Exception as e:
        print(f"Hourly error: {e}")
        return go.Figure(), "⏰ Error loading data"


@app.callback(Output("country-history", "figure"), Input("country-dropdown", "value"))
def update_history(country):
    try:
        country_data = df[df["country"] == country].copy()
        country_data["timestamp"] = pd.to_datetime(country_data["timestamp"])
        monthly = country_data.groupby(country_data["timestamp"].dt.to_period("M"))["outage"].mean().reset_index()
        monthly["timestamp"] = monthly["timestamp"].astype(str)

        fig = px.line(monthly, x="timestamp", y="outage",
                      labels={"outage": "Outage Rate", "timestamp": "Month"},
                      color_discrete_sequence=["#00d4ff"])
        fig.update_layout(
            paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
            font_color="white", height=300,
            xaxis=dict(gridcolor="#333"),
            yaxis=dict(gridcolor="#333"),
            margin=dict(l=0, r=0, t=10, b=0)
        )
        return fig
    except Exception as e:
        print(f"History error: {e}")
        return go.Figure()


@app.callback(Output("top-countries", "figure"), Input("country-dropdown", "value"))
def update_top_countries(_):
    try:
        top = df.groupby("country")["outage"].mean().reset_index()
        top.columns = ["country", "outage_rate"]
        top["outage_pct"] = (top["outage_rate"] * 100).round(1)
        top = top.nlargest(15, "outage_pct")

        fig = px.bar(top, x="outage_pct", y="country", orientation="h",
                     color="outage_pct",
                     color_continuous_scale=["#00ff88", "#ffaa00", "#ff4444"],
                     labels={"outage_pct": "Outage Rate %", "country": ""})
        fig.update_layout(
            paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
            font_color="white", height=400, showlegend=False,
            xaxis=dict(gridcolor="#333"),
            yaxis=dict(gridcolor="#333"),
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        return fig
    except Exception as e:
        print(f"Top countries error: {e}")
        return go.Figure()


@app.callback(Output("feature-importance", "figure"), Input("country-dropdown", "value"))
def update_importance(_):
    try:
        fig = px.bar(importance, x="importance", y="feature", orientation="h",
                     color="importance",
                     color_continuous_scale=["#1a1a2e", "#00d4ff"],
                     labels={"importance": "Importance", "feature": ""})
        fig.update_layout(
            paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
            font_color="white", height=300, showlegend=False,
            xaxis=dict(gridcolor="#333"),
            yaxis=dict(gridcolor="#333"),
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        return fig
    except Exception as e:
        print(f"Feature importance error: {e}")
        return go.Figure()


if __name__ == "__main__":
    print("Dashboard running at http://localhost:8050")
    app.run(debug=True)