
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
from pmdarima import AutoARIMA
from pmdarima.arima import ARIMA
from arch import arch_model
import warnings
import requests
from scipy.signal import periodogram
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import norm
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import skew
import matplotlib.patches as mpatches
from statsmodels.tsa.stattools import ccf
from scipy import stats
import shutil
import matplotlib.dates as mdates

warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=ConvergenceWarning)

def get_csv(url, local_filename):
  full_path = os.path.join("dados", local_filename)
  try:
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    with open(full_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"File '{local_filename}' downloaded successfully.")
  except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")

def recreate_dir(path):
    # se a pasta já existe, apaga ela com tudo dentro
    if os.path.exists(path):
        shutil.rmtree(path)
    # recria a pasta vazia
    os.makedirs(path)


recreate_dir("dados")
recreate_dir("figs")
recreate_dir("resultados_modelos")
recreate_dir("resultado_metricas")
recreate_dir("resultados_figuras")



file_url_dengue = "https://huggingface.co/datasets/JeanDias/InfoDengue_Belem/resolve/main/dengue_34-1.csv"
local_filename_dengue = "dengue.csv"
get_csv(file_url_dengue, local_filename_dengue)
file_url_precip = "https://huggingface.co/datasets/JeanDias/InfoDengue_Belem/resolve/main/prec_Belem_2010_2024.csv"
local_filename_precip = "prec.csv"
get_csv(file_url_precip, local_filename_precip)

sns.set(style="whitegrid")  # estilo Seaborn

def plot_kde_dengue(original, transformed):
    fig, axes = plt.subplots(1, 2, figsize=(14,5))

    # Calcular skewness
    skew_orig = skew(original)
    skew_log = skew(transformed)

    # KDE da série original
    sns.kdeplot(original, ax=axes[0], fill=True, color="skyblue", linewidth=2)
    axes[0].set_title("Distribuição original (a)", fontsize=12)
    axes[0].set_xlabel("Casos")
    axes[0].set_ylabel("Densidade")
    axes[0].text(0.95, 0.95, f"Assímetria: {skew_orig:.2f}",
                 horizontalalignment='right', verticalalignment='top',
                 transform=axes[0].transAxes, fontsize=11, color="darkblue")

    # KDE da série log transformada
    sns.kdeplot(transformed, ax=axes[1], fill=True, color="salmon", linewidth=2)
    axes[1].set_title("Distribuição transformada (b)", fontsize=12)
    axes[1].set_xlabel("log1p(Casos)")
    axes[1].set_ylabel("Densidade")
    axes[1].text(0.95, 0.95, f"Assímetria: {skew_log:.2f}",
                 horizontalalignment='right', verticalalignment='top',
                 transform=axes[1].transAxes, fontsize=11, color="darkred")

    plt.tight_layout()
    plt.savefig("./figs/distribuicao_casos_sns_skew.png", dpi=300)
    plt.show()

def plot_periodogram(log_transformed, fs=1, max_lag=52):
    freqs, power = periodogram(log_transformed, fs=fs)
    periods = 1 / freqs
    mask = (periods <= (max_lag) + 1) & (periods > 1)

    top_idx = np.argsort(power[mask])[-3:][::-1]
    top_periods = periods[mask][top_idx]
    top_powers = power[mask][top_idx]

    plt.figure(figsize=(10,5))
    sns.lineplot(x=periods[mask], y=power[mask], color="steelblue", linewidth=2)
    sns.scatterplot(x=top_periods, y=top_powers, color="red", s=80, zorder=5, label="Top 3 picos")

    for p, pw in zip(top_periods, top_powers):
        plt.text(p, pw*1.05, f"{round(p)}", fontsize=10, ha="center", va="bottom")

    plt.ylim(0, power[mask].max()*1.2)
    plt.title("Periodograma (log transformado)", fontsize=12)
    plt.xlabel("Período (semanas)")
    plt.ylabel("Densidade espectral de potência")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./figs/periodograma_casos_sns.png", dpi=300)
    plt.show()

    return top_periods


def select_top_corr_vars(df, target, quantile=0.75):
    corr = df.corrwith(target).abs()
    threshold = corr.quantile(quantile)
    top_vars = corr[corr >= threshold].index.tolist()
    return top_vars


def fourie_pca_data(df, top_periods, target, quantile=0.75):
    exogenous_variables = df.copy()

    t = np.arange(len(exogenous_variables))
    for i, period in enumerate(top_periods, 1):
        exogenous_variables[f'sin_{i}'] = np.sin(2 * np.pi * t / period)
        exogenous_variables[f'cos_{i}'] = np.cos(2 * np.pi * t / period)

    top_vars = select_top_corr_vars(df.drop(columns=[df.columns[0]]), target, quantile=quantile)

    # PCA apenas nas colunas top
    exogenous_variables = exogenous_variables.dropna()
    pca = PCA(n_components=0.95)
    pca_features = pca.fit_transform(exogenous_variables[top_vars])
    column_names = [f'pca_{i+1}' for i in range(pca_features.shape[1])]
    pca_df = pd.DataFrame(pca_features, columns=column_names, index=exogenous_variables.index)

    # Concatenar Fourier + PCA
    exogenous_variables_all = pd.concat([exogenous_variables.drop(columns=top_vars), pca_df], axis=1)
    return exogenous_variables_all

# --- Função de pré-processamento ---
def preprocess_exog(df, boxcox_vars=None, standard_vars=None):
    df_proc = df.copy()
    if boxcox_vars:
        for var in boxcox_vars:
            df_proc[var] = stats.boxcox(df_proc[var])[0]
    if standard_vars:
        std_model = StandardScaler()
        for var in standard_vars:
            df_proc[var] = std_model.fit_transform(df_proc[var].to_frame())
    return df_proc

# --- Função para calcular CCF e melhores lags ---
def compute_best_lags(df_exog, y, max_lag=52):
    ccf_all = pd.concat([
        pd.DataFrame({
            'variable': name,
            'lag': np.arange(max_lag+1),
            'correlation': ccf(series, y, adjusted=False)[:max_lag+1],
            'abs_corr': np.abs(ccf(series, y, adjusted=False)[:max_lag+1])
        })
        for name, series in df_exog.items()
    ], ignore_index=True)

    best_lags = ccf_all.loc[ccf_all.groupby("variable")["abs_corr"].idxmax()].sort_values("abs_corr", ascending=False)
    return ccf_all, best_lags

# --- Função para plotar CCF com melhores lags ---
def plot_ccf(ccf_all, best_lags, title="CCF entre Y e variáveis exógenas"):
    plt.figure(figsize=(12,6))
    sns.lineplot(data=ccf_all, x="lag", y="correlation", hue="variable")
    for _, r in best_lags.iterrows():
        plt.scatter(
            r["lag"], r["correlation"],
            s=120, marker="o", edgecolors="k", linewidth=1.2,
            label=f"Best lag {r['variable']} = {int(r['lag'])}"
        )
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Lag (semanas)")
    plt.ylabel("Correlação cruzada")
    plt.legend(title="Variável", bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    plt.savefig("./figs/ccf_exog.png", dpi=300)
    plt.show()

# --- Função para construir df_reg com melhores lags ---
def build_regression_df(y, df_exog, best_lags):
    df_reg = y.to_frame()
    for _, r in best_lags.iterrows():
        s = df_exog[r['variable']]
        df_reg[f"{r['variable']}_lag{int(r['lag'])}"] = s.shift(int(r['lag']))
    return df_reg

def get_correlation_map(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlação das Covariáveis")
    plt.savefig("./figs/corr_matrix.png", dpi=300)
    plt.show()

# -----------------------------
# Funções auxiliares
# -----------------------------
def train_arima(y_train, X_train):
    """Treina AutoARIMA com dados exógenos."""
    forecaster = AutoARIMA(
        start_p=0, max_p=5,
        start_q=0, max_q=5,
        d=None, D=None,
        trace=True,
        error_action="ignore",
        information_criterion="bic",
        suppress_warnings=True,
        stepwise=False,
        n_jobs=-1,
        n_fits=100,
        random=True
    )
    model = forecaster.fit(y=y_train, X=X_train)
    return model


def get_resid(model, y_train, X_train):
    """Calcula resíduos in-sample do ARIMA."""
    y_pred_in_sample = model.predict_in_sample(X=X_train)
    residuals = y_train - y_pred_in_sample
    return residuals


def train_garch(residuals, p=1, q=1, o=0, vol='EGARCH', mean='Zero'):
    """Treina GARCH nos resíduos."""
    garch_model = arch_model(residuals, mean=mean, vol=vol, dist="normal", p=p, q=q, o=o, rescale=False)
    garch_model = garch_model.fit(update_freq=0,disp="off")
    return garch_model


def find_best_garch(residuals, max_p=4, max_q=4, max_o=4, vol='EGARCH'):
    """Busca melhor modelo GARCH baseado em BIC e distribuição de resíduos."""
    best_bic = np.inf
    best_p, best_q, best_o, best_mean = None, None, None, None
    means = ["Zero"]

    for mean in means:
      for o in range(1, max_o+1):
          for p in range(1, max_p+1):
              for q in range(1, max_q+1):
                  try:
                      if vol == 'EGARCH':
                        garch_model = arch_model(
                            residuals,
                            mean=mean,
                            vol=vol,
                            dist="normal",
                            p=p,
                            q=q,
                            o=o,
                            rescale=False
                        )
                      elif vol == 'GARCH':
                        garch_model = arch_model(
                            residuals,
                            mean=mean,
                            vol=vol,
                            dist="normal",
                            p=p,
                            q=q,
                            rescale=False
                        )
                      garch_fit = garch_model.fit(update_freq=0, disp="off")
                      if garch_fit.bic < best_bic:
                          best_bic = garch_fit.bic
                          best_p, best_q, best_o, best_mean = p, q, o, mean

                  except Exception as e:
                      print(f"Falhou p={p}, q={q}, o={o}, mean={mean}: {e}")
                      continue
    print(f"Best BIC: {best_bic}")
    print(f"Best p: {best_p}")
    print(f"Best q: {best_q}")
    print(f"Best o: {best_o}")
    print(f"Best mean: {best_mean}")
    return best_p, best_q, best_o, best_mean

def predict(arima_model, garch_model,horizon, X_test):
  y_pred = arima_model.predict(n_periods=horizon, X=X_test.iloc[:horizon])
  garch_forecast = garch_model.forecast(horizon=horizon)
  predicted_std = np.sqrt(garch_forecast.variance.values[-1, :])
  return y_pred, predicted_std


# -----------------------------
# Pipeline completo ARIMAX + EGARCH
# -----------------------------
def train_forecast_pipeline(y_train, X_train, X_test, horizon,
                            use_garch=True, vol='EGARCH'):
    arima_model = train_arima(y_train, X_train)
    residuals = get_resid(arima_model, y_train, X_train)[1:]
    print("Resíduos do ARIMA:")
    print(residuals)
    print("========================")

    if use_garch:
        best_p, best_q, best_o, best_mean = find_best_garch(residuals, vol=vol)
        garch_model = train_garch(residuals, p=best_p, q=best_q, o=best_o, mean=best_mean, vol=vol)
        y_pred, predicted_std = predict(arima_model, garch_model, horizon, X_test)
    else:
        y_pred = arima_model.predict(n_periods=horizon, X=X_test)
        predicted_std = np.std(residuals)

    return y_pred, arima_model, predicted_std, residuals

# -----------------------------
# Função geral para avaliação
# -----------------------------
def avaliar_modelo(path_csv, titulo):
    # Carregar resultados
    df = pd.read_csv(path_csv)
    observado = df['Observado']
    forecast = df['Forecast']
    upper = df['Upper']
    lower = df['Lower']

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(14, 6))
    plt.plot(observado, label="Observado", color="black")
    plt.plot(forecast, label="Forecast", color="blue")
    plt.fill_between(forecast.index, upper, lower,
                     color="blue", alpha=0.2, label="PI 95%")
    plt.title(f"Rolling Forecast ( {titulo} )")
    plt.legend()
    plt.show()

    # -----------------------------
    # Métricas
    # -----------------------------
    y_true = observado.copy()
    y_pred = forecast.copy()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    r = np.corrcoef(y_true, y_pred)[0, 1]
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    median_width = np.median(upper - lower)
    mean_width = np.mean(upper - lower)

    # Printar resultados
    print(f"Modelo: {titulo}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.3f}")
    print(f"Corr: {r:.3f}")
    print(f"Coverage PI: {coverage*100:.1f}%")
    print(f"Median Width PI: {median_width:.3f}")
    print(f"Mean Width PI: {mean_width:.3f}")
    print("-"*40)

    return {
        "Modelo": titulo,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "Corr": r,
        "Coverage": coverage,
        "Median Width": median_width,
        "Mean Width": mean_width
    }


# ------------------------------------------
# Função geral para Treinamento e Inferência
# ------------------------------------------

def run_rolling_forecast(y_train, y_test, X_train, X_test, parameters, horizon, z, split_name):
    rolling_preds, rolling_upper, rolling_lower, rolling_indices = [], [], [], []

    for i in range(horizon):
        print(f'Split {split_name} | Passo Atual: {i+1}/{horizon}')

        # Janela até o ponto atual
        y_window = pd.concat([y_train, y_test.iloc[:i]])
        X_window = pd.concat([X_train, X_test.iloc[:i]])
        X_next = X_test.iloc[[i]]

        # ---------- ARIMA + EGARCH/GARCH ----------
        y_pred, arima_model, predicted_std, residuals = train_forecast_pipeline(
            y_window, X_window, X_next, horizon=1,
            use_garch=parameters[0],
            vol=parameters[1]
        )
        print('=======================')
        print("pred")
        print(y_pred)
        print("std")
        print(predicted_std)
        print('=======================')
        # ---------- Desfazer log1p ----------
        y_final_pred = np.expm1(y_pred + 0.5*predicted_std**2)
        upper = np.expm1(y_pred + z * predicted_std)
        lower = np.expm1(y_pred - z * predicted_std)

        print(f'Previsto: {y_final_pred[0]:.2f} | Observado: {np.expm1(y_test.iloc[i]):.2f}')
        print(f'PI 95%: [{lower[0]:.2f}, {upper[0]:.2f}]')
        print('-----------------------')
        print()

        # ---------- Armazenar ----------
        rolling_preds.append(y_final_pred)
        rolling_upper.append(upper)
        rolling_lower.append(lower)
        rolling_indices.append(y_test.index[i])

    # -----------------------------
    # Converter em séries temporais
    # -----------------------------
    rolling_preds_flat = [val.iloc[0] if isinstance(val, pd.Series) else val for val in rolling_preds]
    rolling_upper_flat = [val.iloc[0] if isinstance(val, pd.Series) else val for val in rolling_upper]
    rolling_lower_flat = [val.iloc[0] if isinstance(val, pd.Series) else val for val in rolling_lower]

    rolling_forecast = pd.Series(rolling_preds_flat, index=rolling_indices)
    rolling_upper_series = pd.Series(rolling_upper_flat, index=rolling_indices)
    rolling_lower_series = pd.Series(rolling_lower_flat, index=rolling_indices)

    resultados = pd.DataFrame({
        'Modelo_1': 'ARIMAX',
        'Modelo_2': parameters[1],
        'Observado': round(np.expm1(y_test)),
        'Forecast': rolling_forecast,
        'Upper': rolling_upper_series,
        'Lower': rolling_lower_series
    })

    # Nome do CSV pelo ano do split
    resultados.to_csv(f'./resultados_modelos/resultados_{split_name}_UseGarch={parameters[0]}_vol={parameters[1]}.csv')


# -----------------------------
# Carregar CSVs e extrair info
# -----------------------------
def load_forecast_results(result_dir="./resultados_modelos/"):
    csv_files = [f for f in os.listdir(result_dir) if f.endswith(".csv")]
    df_list = []

    for file in csv_files:
        path = os.path.join(result_dir, file)
        df = pd.read_csv(path, parse_dates=["Unnamed: 0"])
        df = df.rename(columns={"Unnamed: 0": "data"})

        # Extrair split, use_garch e tipo de vol do nome do arquivo
        parts = file.replace("resultados_", "").replace(".csv", "").split("_")
        split_year = parts[0]
        use_garch = parts[1].split("=")[1]
        vol_type = parts[2].split("=")[1]

        df["split"] = split_year
        df["use_garch"] = use_garch
        df["vol"] = vol_type

        df_list.append(df)

    df_all = pd.concat(df_list, ignore_index=True)
    return df_all

# -----------------------------
# Plotar grade de forecasts
# -----------------------------
def plot_forecast_grid(df_all, filename="resultados_forecast.png"):
    sns.set(style="whitegrid")
    splits_order = sorted(df_all['split'].unique())
    models_order = sorted(df_all['vol'].unique())

    fig, axes = plt.subplots(len(splits_order), len(models_order), figsize=(18,12))

    for i, split in enumerate(splits_order):
        for j, vol_model in enumerate(models_order):
            ax = axes[i, j]
            df_plot = df_all[(df_all['split'] == split) & (df_all['vol'] == vol_model)]

            ax.plot(df_plot['data'], df_plot['Observado'], color="black", label="Observado")
            ax.plot(df_plot['data'], df_plot['Forecast'], color="red", label="Previsto")
            ax.fill_between(df_plot['data'], df_plot['Lower'], df_plot['Upper'], color="red", alpha=0.2)

            # Limites Y independentes
            y_min = df_plot[['Observado', 'Forecast', 'Lower']].min().min() * 0.95
            y_max = df_plot[['Observado', 'Forecast', 'Upper']].max().max() * 1.05
            ax.set_ylim(y_min, y_max)

            ax.set_title(f"Split {split} - {vol_model}")
            if j == 0:
                ax.set_ylabel("Casos")

            # Ticks X formatados
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.tick_params(axis='x', labelrotation=30, labelsize=10, bottom=True, labelbottom=True)

    # Legenda global
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join('resultados_figuras',filename), dpi=300)
    plt.show()

# ================================
# 1. Carregar dados
# ================================
# Série dengue
dengue_path = "./dados/dengue.csv"
serie_dengue = pd.read_csv(dengue_path).sort_values("data_iniSE")
serie_dengue.index = pd.to_datetime(serie_dengue['data_iniSE'])
serie_dengue = serie_dengue[['casprov','tempmin','umidmax','umidmed','umidmin','tempmed','tempmax']]
serie_dengue = serie_dengue.resample("W").mean().interpolate()

# Transformar variável alvo
dengue = np.log1p(serie_dengue['casprov'])

# Série chuva
chuva = pd.read_csv("./dados/prec.csv")
chuva.index = pd.to_datetime(chuva['date'])
chuva = chuva['precipitation_sum'].resample("W").mean().interpolate()

# DataFrame de variáveis exógenas
variables = {'chuva': chuva, **{c: serie_dengue[c] for c in ['tempmin','tempmed','tempmax','umidmin','umidmed','umidmax']}}
df_teste = pd.DataFrame(variables).resample('W').mean().interpolate()


# --- debug ---
#n_rows = int(len(df_teste) * 0.3)  # 20% do total
#df_teste = df_teste.iloc[:n_rows]

# ================================
# 2. Pré-processamento
# ================================
df_teste_proc = preprocess_exog(
    df_teste,
    boxcox_vars=["chuva",'tempmin','tempmed','umidmin'],
    standard_vars=["tempmax","umidmed","umidmax"]
)

# -----------------------------
# 3. Periodograma da Variavel Alvo
# -----------------------------
top_periods = plot_periodogram(dengue, max_lag=60)

# ================================
# 4. Calcular CCF e melhores lags
# ================================
ccf_all, best_lags = compute_best_lags(df_teste_proc, dengue, max_lag=60)

# ================================
# 5. Construir df_reg pronto para regressão/ARIMAX
# ================================
df_reg = build_regression_df(dengue, df_teste_proc, best_lags)
df_reg = fourie_pca_data(df_reg, top_periods,df_reg['casprov'],quantile=0.5)

# ================================
# 6. Plotar CCF e destacar melhores lags
# ================================
plot_ccf(ccf_all, best_lags)

# -----------------------------
# 10. Construção X
# -----------------------------
y = df_reg['casprov']
x = df_reg.drop(columns=['casprov'])
get_correlation_map(df_reg)


# -----------------------------
# 11. Split temporal
# -----------------------------
# Dividir em 3 splits
splits = np.array_split(y.index, 3)

plt.figure(figsize=(12,5))
sns.lineplot(x=y.index, y=np.expm1(y), label="y", color="black")

for i, split_idx in enumerate(splits, 1):
    split_start, split_end = split_idx[0], split_idx[-1]
    y_split = y.loc[split_start:split_end]

    n_split = len(y_split)
    train_size = int(n_split * 0.7)
    train_end = y_split.index[train_size-1]
    test_start = y_split.index[train_size]

    # treino acumulado = do início até o fim do treino atual
    plt.axvspan(y.index[0], train_end, color="skyblue", alpha=0.2)

    # teste = só o bloco final do split atual
    plt.axvspan(test_start, split_end, color="navajowhite", alpha=0.3)

    # cortes entre splits
    if i < 3:
        plt.axvline(split_end, color="red", linestyle="--", alpha=0.7)
        plt.text(split_end, 200, f"Split {i}→{i+1}", rotation=90,
                 verticalalignment="bottom", horizontalalignment="right")

# legendas
train_patch = mpatches.Patch(color='skyblue', alpha=0.2, label='Treino acumulado')
test_patch = mpatches.Patch(color='navajowhite', alpha=0.3, label='Teste atual')
plt.legend(handles=[train_patch, test_patch], loc='upper left')

plt.ylim(0, np.expm1(y).max()*1.1)
plt.xlabel("Data")
plt.ylabel("Casos de Dengue (Escala Original)")
plt.title("Divisão temporal com treino acumulado e teste por split")
plt.savefig("divisao_acumulada.png", dpi=300)
plt.show()



# -----------------------------
# 12.Configuração
# -----------------------------
z = norm.ppf(0.975)  # IC 95%
use_garch = [False, True, True, ]
vol = [None, 'GARCH', 'EGARCH']


# -----------------------------
# 7. Rolling Forecast CV
# -----------------------------

y_accum = pd.Series(dtype=float)
X_accum = pd.DataFrame(dtype=float)

for i, split_idx in enumerate(splits, 1):
    split_start, split_end = split_idx[0], split_idx[-1]
    y_split = y.loc[split_start:split_end]
    X_split = x.loc[split_start:split_end]

    # Definir ponto de corte treino/teste dentro do split atual
    train_size = int(len(y_split) * 0.7)
    y_train_part = y_split.iloc[:train_size]
    X_train_part = X_split.iloc[:train_size]
    y_test = y_split.iloc[train_size:]
    X_test = X_split.iloc[train_size:]

    # Treino acumulado + parte de treino do split atual
    y_train_full = pd.concat([y_accum, y_train_part])
    X_train_full = pd.concat([X_accum, X_train_part])

    # Garantir alinhamento semanal
    y_train_full = y_train_full.resample('W').mean()
    y_test = y_test.resample('W').mean()
    X_train_full = X_train_full.resample('W').mean()
    X_test = X_test.resample('W').mean()

    horizon = len(y_test)
    split_name = str(y_test.index[0].year)

    for parameters in list(zip(use_garch, vol)):
        print(y_test.index[0])
        print(split_name)
        run_rolling_forecast(y_train_full, y_test, X_train_full, X_test, parameters, horizon, z, split_name)

    # Atualiza histórico acumulado com o split inteiro (treino+teste)
    y_accum = pd.concat([y_accum, y_split])
    X_accum = pd.concat([X_accum, X_split])


# -----------------------------
# Resultado Final
# -----------------------------
resultados = []
csvs = sorted(os.listdir("./resultados_modelos/"))
for arquivo in csvs:
    resultado_de_cada_modelo = avaliar_modelo(os.path.join("./resultados_modelos", arquivo), arquivo.replace(".csv", ""))
    resultados.append(resultado_de_cada_modelo)

# -----------------------------
# Consolidar resultados
# -----------------------------
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("./resultado_metricas/resultados_consolidados.csv", index=False)

# -----------------------------
# Plotar todos os resultados
# -----------------------------
df_all = load_forecast_results(result_dir="./resultados_modelos/")
plot_forecast_grid(df_all)

pd.read_csv("./resultado_metricas/resultados_consolidados.csv")

