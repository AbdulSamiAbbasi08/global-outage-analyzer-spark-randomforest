import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import joblib
import numpy as np
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Loading data and model...")

# Load cleaned data and model
df = pd.read_csv("data/cleaned_outage_data.csv")
model = joblib.load("models/rf_model.pkl")
metrics = pd.read_csv("models/metrics.csv")
importance = pd.read_csv("models/feature_importance.csv")

# Country list for dropdown
countries = sorted(df["country"].unique().tolist())

# App setup
app = dash.Dash(__name__)
app.title = "Global Internet Outage Analyzer"

# ── Layout ──
app.layout = html.Div(style={"fontFamily": "Arial", "backgroundColor": "#0f1117", "color": "white", "minHeight": "100vh", "padding": "20px"}, children=[

    # Header
    html.Div([
        html.H1("🌐 Global Internet Outage Analyzer", style={"textAlign": "center", "color": "#00d4ff", "marginBottom": "5px"}),
        html.P("Powered by Apache Spark + Random Forest | Big Data Analytics Project", style={"textAlign": "center", "color": "#888", "marginTop": "0"}),
    ]),

    # Model metrics bar
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

    # Country selector + prediction
    html.Div([
        html.Div([
            html.H3("🔍 Country Analysis", style={"color": "#00d4ff"}),
            html.Label("Select Country:", style={"color": "#888"}),
            dcc.Dropdown(
                id="country-dropdown",
                options=[{"label": c, "value": c} for c in countries],
                value="Pakistan",
                style={"backgroundColor": "#0f1117", "color": "black"}
            ),
            html.Br(),
            html.Label("Select Hour (0-23):", style={"color": "#888"}),
            dcc.Slider(
                id="hour-slider", min=0, max=23, step=1, value=12,
                marks={i: {"label": str(i), "style": {"color": "white"}} for i in range(0, 24, 3)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Br(),
            html.Label("Weather Condition:", style={"color": "#888"}),
            dcc.Dropdown(
                id="weather-dropdown",
                options=[
                    {"label": "Clear", "value": 0},
                    {"label": "Cloudy", "value": 3},
                    {"label": "Rain", "value": 1},
                    {"label": "Storm", "value": 2},
                    {"label": "Snow", "value": 4},
                ],
                value=0,
                style={"backgroundColor": "#0f1117", "color": "black"}
            ),
            html.Br(),
            html.Div(id="prediction-output"),
        ], style={"flex": "1", "backgroundColor": "#1a1a2e", "padding": "20px", "borderRadius": "10px"}),

        # Outage history chart
        html.Div([
            html.H3("📈 Outage History", style={"color": "#00d4ff"}),
            dcc.Graph(id="country-history"),
        ], style={"flex": "2", "backgroundColor": "#1a1a2e", "padding": "20px", "borderRadius": "10px"}),

    ], style={"display": "flex", "gap": "15px", "marginBottom": "20px"}),

    # Bottom charts
    html.Div([
        html.Div([
            html.H3("🏆 Top 15 Countries by Outage Rate", style={"color": "#00d4ff"}),
            dcc.Graph(id="top-countries"),
        ], style={"flex": "1", "backgroundColor": "#1a1a2e", "padding": "20px", "borderRadius": "10px"}),

        html.Div([
            html.H3("🧠 Feature Importance", style={"color": "#00d4ff"}),
            dcc.Graph(id="feature-importance"),
        ], style={"flex": "1", "backgroundColor": "#1a1a2e", "padding": "20px", "borderRadius": "10px"}),
    ], style={"display": "flex", "gap": "15px"}),
])


# ── Callbacks ──

@app.callback(Output("world-map", "figure"), Input("country-dropdown", "value"))
def update_map(_):
    try:
        # 2-letter to 3-letter ISO code mapping
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
            country_stats,
            locations="iso3",
            color="outage_pct",
            hover_name="country",
            locationmode="ISO-3",
            color_continuous_scale=["#00ff88", "#ffaa00", "#ff4444"],
            range_color=[0, 80],
            labels={"outage_pct": "Risk %"},
        )

        fig.update_geos(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#888",
            showland=True,
            landcolor="#3a3a3a",
            showocean=True,
            oceancolor="#1a1a2e",
            projection_type="equirectangular",
        )

        fig.update_layout(
            paper_bgcolor="#1a1a2e",
            font_color="white",
            margin=dict(l=0, r=0, t=0, b=0),
            height=420,
            coloraxis_colorbar=dict(
                title=dict(text="Risk %", font=dict(color="white")),
                tickfont=dict(color="white"),
            ),
        )

        return fig
    except Exception as e:
        print(f"Map error: {e}")
        return go.Figure()


@app.callback(Output("prediction-output", "children"),
              Input("country-dropdown", "value"),
              Input("hour-slider", "value"),
              Input("weather-dropdown", "value"))
def predict_outage(country, hour, weather_index):
    try:
        country_data = df[df["country"] == country]
        if len(country_data) == 0:
            return html.P("No data available")

        avg = country_data.mean(numeric_only=True)

        features = np.array([[
            hour,
            3,
            6,
            avg["bgp_signal"],
            avg["active_probing"],
            avg["traffic_drop_pct"],
            avg["latency_ms"],
            weather_index,
            avg["country_index"],
            avg["base_outage_prob"],
        ]])

        prob = model.predict_proba(features)[0][1]
        pct = round(prob * 100, 1)

        if pct < 30:
            color = "#00ff88"
            risk = "🟢 LOW RISK"
        elif pct < 60:
            color = "#ffaa00"
            risk = "🟡 MEDIUM RISK"
        else:
            color = "#ff4444"
            risk = "🔴 HIGH RISK"

        return html.Div([
            html.H2(risk, style={"color": color, "textAlign": "center", "margin": "10px 0"}),
            html.H1(f"{pct}%", style={"color": color, "textAlign": "center", "fontSize": "60px", "margin": "0"}),
            html.P("Outage Probability", style={"textAlign": "center", "color": "#888"}),
            html.Hr(style={"borderColor": "#333"}),
            html.P(f"Country: {country}", style={"color": "#ccc", "margin": "5px 0"}),
            html.P(f"Hour: {hour}:00", style={"color": "#ccc", "margin": "5px 0"}),
            html.P(f"Avg Latency: {avg['latency_ms']:.0f}ms", style={"color": "#ccc", "margin": "5px 0"}),
            html.P(f"BGP Signal: {avg['bgp_signal']:.1f}", style={"color": "#ccc", "margin": "5px 0"}),
        ])
    except Exception as e:
        print(f"Prediction error: {e}")
        return html.P("Prediction error", style={"color": "red"})


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
            font_color="white", height=400, showlegend=False,
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