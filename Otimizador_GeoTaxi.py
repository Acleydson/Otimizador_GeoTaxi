import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import utm
import numpy as np
import matplotlib.pyplot as plt



# =========================
# Configuração da Página
# =========================
st.set_page_config(layout="wide", page_title="Otimizador GeoTaxi (Dissertação)", page_icon="🚒")

# =========================
# Estilos CSS
# =========================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #e4ecf5 100%);
}

.hero-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    color: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 18px 40px rgba(0,0,0,0.15);
    margin-bottom: 18px;
}

.hero-box h1 {
    font-size: 1.8rem;
    font-weight: 700;
}

.panel-box {
    background: rgba(255,255,255,0.95);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}

.result-box {
    background: white;
    border-radius: 16px;
    padding: 16px;
    border-left: 6px solid #2563eb;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 12px;
}

.stButton>button {
    background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
    color: white;
    border-radius: 12px;
    font-weight: 600;
    padding: 0.7rem 1.2rem;
    border: none;
    transition: 0.2s ease-in-out;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(37,99,235,0.3);
}

.stMetric {
    background: white;
    padding: 12px;
    border-radius: 12px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.08);
}
            
.legend-box {
    background: rgba(255, 255, 255, 0.97);
    border-radius: 16px;
    padding: 14px;
    margin-top: 12px;
    font-size: 14px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.08);
}


</style>
""", unsafe_allow_html=True)



# =========================
# Funções Auxiliares (dados)
# =========================
def limpar_e_normalizar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza colunas e limpa dados numéricos."""
    mapa_sinonimos = {
        "lat": ["latitude", "lat", "y", "lat_dec", "nr_latitude"],
        "lon": ["longitude", "lon", "long", "x", "lon_dec", "nr_longitude"],
        "bairro": ["bairro", "nome", "local", "ponto", "id", "cidade", "endereco"],
        "peso": ["peso", "ocorrencias", "demandas", "risco", "weight", "qtd"],
    }

    novo_mapa = {}
    colunas_lower = {col: col.lower() for col in df.columns}

    for padrao, sinonimos in mapa_sinonimos.items():
        for col_real, col_lower in colunas_lower.items():
            if any(sin in col_lower for sin in sinonimos):
                if padrao not in novo_mapa.values():
                    novo_mapa[col_real] = padrao

    if novo_mapa:
        df = df.rename(columns=novo_mapa)

    if "bairro" not in df.columns:
        df["bairro"] = [f"Ponto {i + 1}" for i in range(len(df))]
    if "peso" not in df.columns:
        df["peso"] = 1

    for col in ["lat", "lon", "peso"]:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["lat", "lon"])
    df["peso"] = df["peso"].fillna(1)
    df.loc[df["peso"] <= 0, "peso"] = 1
    return df

def limpar_e_normalizar_dados_cartesiano(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza colunas para modo cartesiano e limpa dados numéricos (x/y).
    Importante: nunca lança KeyError se x/y não existirem (retorna DF vazio).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["bairro", "x", "y", "peso"])

    mapa_sinonimos = {
        "x": ["x", "coordx", "easting", "abscissa"],
        "y": ["y", "coordy", "northing", "ordenada"],
        "bairro": ["bairro", "nome", "local", "ponto", "id", "cidade", "endereco"],
        "peso": ["peso", "ocorrencias", "demandas", "risco", "weight", "qtd"],
    }

    novo_mapa = {}
    colunas_lower = {col: str(col).lower() for col in df.columns}

    for padrao, sinonimos in mapa_sinonimos.items():
        for col_real, col_lower in colunas_lower.items():
            if any(sin == col_lower or sin in col_lower for sin in sinonimos):
                if padrao not in novo_mapa.values():
                    novo_mapa[col_real] = padrao

    if novo_mapa:
        df = df.rename(columns=novo_mapa)

    # Se o arquivo não tem x/y, não quebra: devolve DF vazio padronizado
    if not {"x", "y"}.issubset(df.columns):
        out = pd.DataFrame(columns=["bairro", "x", "y", "peso"])
        return out

        # Garantir colunas mínimas
    if "bairro" not in df.columns:
        df["bairro"] = [f"Ponto {i + 1}" for i in range(len(df))]

    if "peso" not in df.columns:
        df["peso"] = 1

    # Se x/y não existirem, cria e devolve DF vazio (evita KeyError e mantém UI estável)
    if "x" not in df.columns:
        df["x"] = np.nan
    if "y" not in df.columns:
        df["y"] = np.nan


    for col in ["x", "y", "peso"]:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Se ainda não houver dados válidos em x/y, retorna vazio sem explodir
    df = df.dropna(subset=["x", "y"], how="any")
    if df.empty:
        return pd.DataFrame(columns=["bairro", "x", "y", "peso"])

    df["peso"] = df["peso"].fillna(1)
    df.loc[df["peso"] <= 0, "peso"] = 1

    return df[["bairro", "x", "y", "peso"]].copy()



def get_utm_data(df: pd.DataFrame):
    """Converte Lat/Lon para UTM e retorna DataFrame enriquecido."""
    if df.empty:
        return None, None, None

    try:
        _, _, zone_num, zone_let = utm.from_latlon(df.iloc[0]["lat"], df.iloc[0]["lon"])
    except Exception:
        return None, None, None

    utm_list = []
    for _, row in df.iterrows():
        e, n, _, _ = utm.from_latlon(
            row["lat"],
            row["lon"],
            force_zone_number=zone_num,
            force_zone_letter=zone_let,
        )
        utm_list.append(
            {
                "x_utm": float(e),
                "y_utm": float(n),
                "peso": float(row["peso"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "bairro": str(row["bairro"]),
            }
        )

    return pd.DataFrame(utm_list), zone_num, zone_let

#------------================++++++++++++++++++++&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

def preparar_dados_para_modelo(df: pd.DataFrame, modo_cartesiano: bool):
    if df is None or df.empty:
        return None, None, None

    if modo_cartesiano:
        # Aceita x/y (cartesiano). Se não existir, falha de forma controlada.
        if not {"x", "y"}.issubset(df.columns):
            return None, None, None

        df_modelo = df.copy()
        df_modelo["x_utm"] = pd.to_numeric(df_modelo["x"], errors="coerce")
        df_modelo["y_utm"] = pd.to_numeric(df_modelo["y"], errors="coerce")

        if "peso" not in df_modelo.columns:
            df_modelo["peso"] = 1
        if "bairro" not in df_modelo.columns:
            df_modelo["bairro"] = [f"Ponto {i + 1}" for i in range(len(df_modelo))]

        df_modelo = df_modelo.dropna(subset=["x_utm", "y_utm"])
        df_modelo["peso"] = pd.to_numeric(df_modelo["peso"], errors="coerce").fillna(1)
        df_modelo.loc[df_modelo["peso"] <= 0, "peso"] = 1

        return df_modelo, None, None

    return get_utm_data(df)

#------------================++++++++++++++++++++&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


# =========================
# UTM -> Lat/Lon robusto
# =========================
def safe_to_latlon(easting, northing, zone_num, zone_let):
    try:
        return utm.to_latlon(float(easting), float(northing), zone_num, zone_let, strict=True)
    except Exception:
        return utm.to_latlon(float(easting), float(northing), zone_num, zone_let, strict=False)


def utm_xy_to_latlon_list(xy_list, zone_num, zone_let):
    out = []
    for x, y in xy_list:
        lat, lon = safe_to_latlon(float(x), float(y), zone_num, zone_let)
        out.append([float(lat), float(lon)])
    return out


# =========================
# Geometria Táxi (L1)
# =========================
def _xy_to_uv(x: np.ndarray, y: np.ndarray):
    return x + y, x - y


def _uv_to_xy(u: np.ndarray, v: np.ndarray):
    return (u + v) / 2.0, (u - v) / 2.0


def mediana_1d(vals: np.ndarray):
    return float(np.median(vals.astype(float)))


def mediana_ponderada_1d(vals: np.ndarray, pesos: np.ndarray):
    ordem = np.argsort(vals)
    vals_o = vals[ordem]
    w_o = pesos[ordem]
    total = float(np.sum(w_o))
    ac = 0.0
    for v_, pw in zip(vals_o, w_o):
        ac += float(pw)
        if ac >= total / 2.0:
            return float(v_)
    return float(vals_o[-1])




def distancia_taxi_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(abs(float(x1) - float(x2)) + abs(float(y1) - float(y2)))

def plot_conjunto_otimo(ax, reg, tipo: str, **kwargs):
    """
    Plota o conjunto ótimo no modo cartesiano.
    - tipo='ponto'    : reg = (x,y) ou [(x,y)]
    - tipo='segmento' : reg = [(x1,y1),(x2,y2)]
    - tipo='regiao'   : reg = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] (polígono)
    """
    if reg is None:
        return

    # normaliza entrada
    if isinstance(reg, tuple) and len(reg) == 2:
        pts = [reg]
    else:
        pts = list(reg)

    if len(pts) == 0:
        return

    if tipo == "ponto":
        x0, y0 = pts[0]
        ax.scatter([x0], [y0], marker="x", s=90, linewidths=2)

    elif tipo == "segmento":
        if len(pts) >= 2:
            (x1, y1), (x2, y2) = pts[0], pts[1]
            ax.plot([x1, x2], [y1, y2], **kwargs)

    elif tipo == "regiao":
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        ax.plot(xs, ys, **kwargs)
        # preenchimento leve (sem forçar cor)
        ax.fill(xs, ys, alpha=0.08)


def inicializar_centros_taxi(df_utm: pd.DataFrame, k: int, usar_pesos: bool):
    pts = df_utm[["x_utm", "y_utm"]].to_numpy(dtype=float)
    w = (
        df_utm["peso"].to_numpy(dtype=float)
        if usar_pesos
        else np.ones(len(df_utm))
    )

    n = len(pts)
    if n == 0:
        return []

    centers = []

    # Primeiro centro: ponderado aleatório
    probs = w / w.sum()
    idx0 = np.random.choice(n, p=probs)
    centers.append(tuple(pts[idx0]))

    for _ in range(1, k):
        dists = np.array(
            [
                min(distancia_taxi_xy(px, py, cx, cy) for cx, cy in centers)
                for px, py in pts
            ]
        )
        weighted_dists = dists * w
        if weighted_dists.sum() == 0:
            idx = np.random.randint(0, n)
        else:
            probs = weighted_dists / weighted_dists.sum()
            idx = np.random.choice(n, p=probs)

        centers.append(tuple(pts[idx]))

    return [(float(cx), float(cy)) for cx, cy in centers]


def atribuir_clusters_taxi(df_utm: pd.DataFrame, centros):
    pts = df_utm[["x_utm", "y_utm"]].to_numpy(dtype=float)
    centers = np.array(centros)

    # Broadcasting vetorizado
    dists = np.abs(pts[:, None, 0] - centers[:, 0]) + np.abs(
        pts[:, None, 1] - centers[:, 1]
    )

    return np.argmin(dists, axis=1)


def calcular_raio_taxi(df_utm: pd.DataFrame, cx: float, cy: float) -> float:
    if df_utm.empty:
        return 0.0
    x = df_utm["x_utm"].to_numpy(dtype=float)
    y = df_utm["y_utm"].to_numpy(dtype=float)
    return float(np.max(np.abs(x - cx) + np.abs(y - cy)))


def soma_distancias_taxi(df_utm: pd.DataFrame, cx: float, cy: float, usar_pesos: bool) -> float:
    if df_utm.empty:
        return 0.0
    x = df_utm["x_utm"].to_numpy(dtype=float)
    y = df_utm["y_utm"].to_numpy(dtype=float)
    d = np.abs(x - cx) + np.abs(y - cy)
    if usar_pesos:
        w = df_utm["peso"].to_numpy(dtype=float)
        return float(np.sum(w * d))
    return float(np.sum(d))


# =========================
# 1-centro (minimax) em L1
# =========================
def one_center_L1_unweighted(df_utm: pd.DataFrame):
    if df_utm.empty:
        return None

    x = df_utm["x_utm"].to_numpy(dtype=float)
    y = df_utm["y_utm"].to_numpy(dtype=float)
    u, v = _xy_to_uv(x, y)

    umin, umax = float(u.min()), float(u.max())
    vmin, vmax = float(v.min()), float(v.max())
    du, dv = (umax - umin), (vmax - vmin)

    R = 0.5 * max(du, dv)

    uL, uU = (umax - R), (umin + R)
    vL, vU = (vmax - R), (vmin + R)

    if uL > uU:
        m = 0.5 * (uL + uU)
        uL = uU = m
    if vL > vU:
        m = 0.5 * (vL + vU)
        vL = vU = m

    u0, v0 = 0.5 * (uL + uU), 0.5 * (vL + vU)
    x0, y0 = _uv_to_xy(np.array([u0]), np.array([v0]))
    x0, y0 = float(x0[0]), float(y0[0])

    eps = 1e-9
    u_len = abs(uU - uL)
    v_len = abs(vU - vL)

    if u_len <= eps and v_len <= eps:
        tipo = "ponto"
        regiao = [(x0, y0)]
    elif u_len <= eps or v_len <= eps:
        tipo = "segmento"
        if u_len <= eps:
            uv_seg = np.array([[uL, vL], [uL, vU]], dtype=float)
        else:
            uv_seg = np.array([[uL, vL], [uU, vL]], dtype=float)
        xs, ys = _uv_to_xy(uv_seg[:, 0], uv_seg[:, 1])
        regiao = [(float(xs[0]), float(ys[0])), (float(xs[1]), float(ys[1]))]
    else:
        tipo = "regiao"
        uv_corners = np.array([[uL, vL], [uU, vL], [uU, vU], [uL, vU]], dtype=float)
        xs, ys = _uv_to_xy(uv_corners[:, 0], uv_corners[:, 1])
        regiao = [(float(xs[i]), float(ys[i])) for i in range(4)]

    info = {
        "tipo": tipo,
        "R_star": float(R),
        "u_interval": (float(uL), float(uU)),
        "v_interval": (float(vL), float(vU)),
    }

    return {"x": x0, "y": y0, "R": float(R), "regiao": regiao, "info": info}


def one_center_L1_weighted(df_utm: pd.DataFrame, mode: str = "multiplicativo", n_iter: int = 60):
    if df_utm.empty:
        return None

    x = df_utm["x_utm"].to_numpy(dtype=float)
    y = df_utm["y_utm"].to_numpy(dtype=float)
    w = df_utm["peso"].to_numpy(dtype=float)
    u, v = _xy_to_uv(x, y)

    if mode not in {"multiplicativo", "divisivo"}:
        raise ValueError("mode deve ser 'multiplicativo' ou 'divisivo'.")

    def feasible(R):
        if mode == "multiplicativo":
            r = R / w
        else:
            r = R * w

        uL = np.max(u - r)
        uU = np.min(u + r)
        vL = np.max(v - r)
        vU = np.min(v + r)
        return (uL <= uU) and (vL <= vU), float(uL), float(uU), float(vL), float(vU)

    lo, hi = 0.0, 1.0
    ok, *_ = feasible(hi)
    guard = 0
    while not ok and guard < 80:
        hi *= 2.0
        ok, *_ = feasible(hi)
        guard += 1
    if not ok:
        return None

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        ok, *_ = feasible(mid)
        if ok:
            hi = mid
        else:
            lo = mid

    R_star = hi
    _, uL, uU, vL, vU = feasible(R_star)

    u0, v0 = 0.5 * (uL + uU), 0.5 * (vL + vU)
    x0, y0 = _uv_to_xy(np.array([u0]), np.array([v0]))
    x0, y0 = float(x0[0]), float(y0[0])

    eps = 1e-9
    u_len = abs(uU - uL)
    v_len = abs(vU - vL)

    if u_len <= eps and v_len <= eps:
        tipo = "ponto"
        regiao = [(x0, y0)]
    elif u_len <= eps or v_len <= eps:
        tipo = "segmento"
        if u_len <= eps:
            uv_seg = np.array([[uL, vL], [uL, vU]], dtype=float)
        else:
            uv_seg = np.array([[uL, vL], [uU, vL]], dtype=float)
        xs, ys = _uv_to_xy(uv_seg[:, 0], uv_seg[:, 1])
        regiao = [(float(xs[0]), float(ys[0])), (float(xs[1]), float(ys[1]))]
    else:
        tipo = "regiao"
        uv_corners = np.array([[uL, vL], [uU, vL], [uU, vU], [uL, vU]], dtype=float)
        xs, ys = _uv_to_xy(uv_corners[:, 0], uv_corners[:, 1])
        regiao = [(float(xs[i]), float(ys[i])) for i in range(4)]

    info = {
        "tipo": tipo,
        "mode": mode,
        "R_star": float(R_star),
        "u_interval": (float(uL), float(uU)),
        "v_interval": (float(vL), float(vU)),
    }

    return {"x": x0, "y": y0, "R": float(R_star), "regiao": regiao, "info": info}


# =========================
# 1-mediana (Weber-Táxi)
# =========================
def one_median_L1(df_utm: pd.DataFrame, usar_pesos: bool):
    if df_utm.empty:
        return None

    xs = df_utm["x_utm"].to_numpy(dtype=float)
    ys = df_utm["y_utm"].to_numpy(dtype=float)

    if usar_pesos:
        w = df_utm["peso"].to_numpy(dtype=float)
        x_med = mediana_ponderada_1d(xs, w)
        y_med = mediana_ponderada_1d(ys, w)
    else:
        x_med = mediana_1d(xs)
        y_med = mediana_1d(ys)

    return float(x_med), float(y_med)


def retangulo_solucoes_weber_2pontos(df_utm: pd.DataFrame):
    if len(df_utm) != 2:
        return None
    x = df_utm["x_utm"].to_numpy(dtype=float)
    y = df_utm["y_utm"].to_numpy(dtype=float)
    xmin, xmax = float(min(x)), float(max(x))
    ymin, ymax = float(min(y)), float(max(y))
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


# =========================
# Losango L1 para PLOT
# =========================
def losango_L1_latlon(
    center_x_utm, center_y_utm, raio_metros, zone_num, zone_let, raio_max_plot=150_000
):
    cx = float(center_x_utm)
    cy = float(center_y_utm)

    r = float(raio_metros)
    if r > raio_max_plot:
        r = float(raio_max_plot)

    offsets = [(r, 0.0), (0.0, r), (-r, 0.0), (0.0, -r)]
    coords = []
    for dx, dy in offsets:
        lat, lon = safe_to_latlon(cx + dx, cy + dy, zone_num, zone_let)
        coords.append([float(lat), float(lon)])
    coords.append(coords[0])
    return coords


# =========================
# p-centro / p-mediana
# =========================
def refinar_clusters_manhattan(
    df_utm: pd.DataFrame,
    k_clusters: int,
    center_mode: str,
    n_iter: int = 20,
    tol: float = 1e-6,  # NOVO
):
    df_utm = df_utm.copy()

    centros = inicializar_centros_taxi(df_utm, k_clusters, usar_pesos=True)
    df_utm["cluster"] = atribuir_clusters_taxi(df_utm, centros)

    for _ in range(n_iter):

        centros_anteriores = centros.copy()
        novos_centros = []

        for i in range(k_clusters):
            cluster_data = df_utm[df_utm["cluster"] == i]

            if cluster_data.empty:
                rand_pt = df_utm.sample(1).iloc[0]
                novos_centros.append(
                    (float(rand_pt["x_utm"]), float(rand_pt["y_utm"]))
                )
                continue

            if center_mode == "nao_ponderado":
                sol = one_center_L1_unweighted(cluster_data)
            else:
                sol = one_center_L1_weighted(cluster_data, mode=center_mode)

            novos_centros.append((float(sol["x"]), float(sol["y"])))

        # Atualiza clusters
        labels_anteriores = df_utm["cluster"].values.copy()
        df_utm["cluster"] = atribuir_clusters_taxi(df_utm, novos_centros)

        # Critério 1 — rótulos iguais
        if np.array_equal(df_utm["cluster"].values, labels_anteriores):
            break

        # Critério 2 — movimento máximo dos centros
        movimento_max = max(
            distancia_taxi_xy(cx, cy, ncx, ncy)
            for (cx, cy), (ncx, ncy) in zip(centros_anteriores, novos_centros)
        )

        if movimento_max < tol:
            break

        centros = novos_centros

    return df_utm, centros


def otimizar_pcentro(
    df,
    velocidade_kmh,
    tortuosidade,
    tempo_max_min,
    center_mode,
    modo_cartesiano=False,
    k_max=None,
):

    df_utm, zn, zl = preparar_dados_para_modelo(df, modo_cartesiano)
    if df_utm is None:
        return None

    velocidade_ms = velocidade_kmh / 3.6
    tempo_s = tempo_max_min * 60.0
    distancia_limite_real = velocidade_ms * tempo_s
    distancia_limite_taxi = distancia_limite_real / float(tortuosidade)

    n = len(df_utm)
    limite_superior = k_max if k_max is not None else min(n, 25)

    for k in range(1, limite_superior + 1):

        df_class, centros_utm = refinar_clusters_manhattan(
            df_utm, k, center_mode=center_mode
        )

        centros_finais = []
        ok = True

        for i, (cx, cy) in enumerate(centros_utm):
            cluster_data = df_class[df_class["cluster"] == i]
            if cluster_data.empty:
                continue

            raio_cluster = calcular_raio_taxi(cluster_data, cx, cy)

            if raio_cluster > distancia_limite_taxi:
                ok = False
                break

            if modo_cartesiano:
                lat_c, lon_c = cx, cy
                poly = None
            else:
                lat_c, lon_c = safe_to_latlon(cx, cy, zn, zl)
                poly = losango_L1_latlon(cx, cy, distancia_limite_taxi, zn, zl)

            centros_finais.append(
                {
                    "id": i,
                    "lat": float(lat_c),
                    "lon": float(lon_c),
                    "x_utm": float(cx),
                    "y_utm": float(cy),
                    "raio_cluster": float(raio_cluster),
                    "cobertura_poly": poly,
                    "pontos_atendidos": int(len(cluster_data)),
                }
            )

        if ok:
            return (centros_finais, df_class, (zn, zl), float(distancia_limite_taxi))

    return None


def refinar_clusters_pmediana(
    df_utm: pd.DataFrame,
    p: int,
    usar_pesos: bool,
    n_iter: int = 30,
    tol: float = 1e-6,  # NOVO
):
    df_local = df_utm.copy()

    centros = inicializar_centros_taxi(df_local, p, usar_pesos=usar_pesos)
    df_local["cluster"] = atribuir_clusters_taxi(df_local, centros)

    for _ in range(n_iter):

        centros_anteriores = centros.copy()
        novos_centros = []

        for i in range(p):
            cluster_data = df_local[df_local["cluster"] == i]

            if cluster_data.empty:
                rand_pt = df_local.sample(1).iloc[0]
                novos_centros.append(
                    (float(rand_pt["x_utm"]), float(rand_pt["y_utm"]))
                )
                continue

            x_med, y_med = one_median_L1(cluster_data, usar_pesos=usar_pesos)
            novos_centros.append((float(x_med), float(y_med)))

        labels_prev = df_local["cluster"].values.copy()
        df_local["cluster"] = atribuir_clusters_taxi(df_local, novos_centros)

        # Critério 1 — rótulos iguais
        if np.array_equal(df_local["cluster"].values, labels_prev):
            break

        # Critério 2 — movimento máximo
        movimento_max = max(
            distancia_taxi_xy(cx, cy, ncx, ncy)
            for (cx, cy), (ncx, ncy) in zip(centros_anteriores, novos_centros)
        )

        if movimento_max < tol:
            break

        centros = novos_centros

    return df_local, centros


def resolver_pmediana(df: pd.DataFrame, p: int, usar_pesos: bool, modo_cartesiano=False):

    df_utm, zn, zl = preparar_dados_para_modelo(df, modo_cartesiano)
    if df_utm is None:
        return None

    df_class, centros_utm = refinar_clusters_pmediana(
        df_utm, p=p, usar_pesos=usar_pesos
    )

    centros_finais = []
    soma_obj = 0.0

    for i, (cx, cy) in enumerate(centros_utm):
        cluster_data = df_class[df_class["cluster"] == i]
        if cluster_data.empty:
            continue

        raio_cluster = calcular_raio_taxi(cluster_data, cx, cy)
        custo_cluster = soma_distancias_taxi(cluster_data, cx, cy, usar_pesos)
        soma_obj += custo_cluster

        if modo_cartesiano:
            lat_c, lon_c = cx, cy
        else:
            lat_c, lon_c = safe_to_latlon(cx, cy, zn, zl)

        centros_finais.append(
            {
                "id": i,
                "lat": float(lat_c),
                "lon": float(lon_c),
                "x_utm": float(cx),
                "y_utm": float(cy),
                "raio_cluster": float(raio_cluster),
                "custo_cluster": float(custo_cluster),
                "pontos_atendidos": int(len(cluster_data)),
            }
        )

    return centros_finais, df_class, (zn, zl), float(soma_obj)


# =========================
# Interface
# =========================
st.markdown("""
<div class="hero-box">
  <h1>🚒 Otimizador GeoTaxi</h1>
  <p>Modelagem em geometria Táxi (L1) para 1-centro, 1-mediana, p-centro e p-mediana.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.subheader("1. Configuração")

    sistema_coord = st.radio(
        "Sistema de Coordenadas:",
        ["Geográficas (lat/lon)", "Cartesianas (x/y)"],
    )
    modo_cartesiano = sistema_coord.startswith("Cartesianas")

# Se o usuário alternar o sistema de coordenadas, descarta resultados anteriores
# (evita KeyError ao tentar plotar dados cartesianos no mapa geográfico e vice-versa)
if "last_modo_cartesiano" not in st.session_state:
    st.session_state["last_modo_cartesiano"] = modo_cartesiano
else:
    if st.session_state["last_modo_cartesiano"] != modo_cartesiano:
        st.session_state.pop("resultado", None)
        st.session_state["last_modo_cartesiano"] = modo_cartesiano

    modo = st.radio(
        "Método:",
        [
            "📍 1-centro (Minimax)",
            "📍 1-mediana (Weber)",
            "⚡ p-centro (múltiplas bases)",
            "⚡ p-mediana (múltiplas bases)",
        ],
    )

    usar_pesos = st.checkbox("Usar pesos da coluna 'peso'", value=True)

    center_mode = "nao_ponderado"
    if ("centro" in modo) and usar_pesos:
        center_mode_ui = st.radio(
            "Modelo de ponderação para minimax (1-centro / p-centro)",
            ["Multiplicativo", "Divisivo"],
            horizontal=True,
        )
        center_mode = "multiplicativo" if center_mode_ui == "Multiplicativo" else "divisivo"

        st.markdown(
            """
<div class="info-box">
<b>Como ler os pesos no minimax:</b><br>
• <b>Multiplicativo</b>: minimiza o pior valor de <code>wᵢ·dᵢ</code>. Peso maior torna o ponto mais exigente.<br>
• <b>Divisivo</b>: minimiza o pior valor de <code>dᵢ/wᵢ</code>. Peso maior 'alivia' a penalização da distância daquele ponto.
</div>
""",
            unsafe_allow_html=True,
        )

    # Parâmetros extras
    velocidade = 40
    tempo_max = 8
    tortuosidade = 1.3
    p_fixado = 3

    if "p-centro" in modo:
        c1, c2 = st.columns(2)
        with c1:
            velocidade = st.slider("Velocidade (km/h)", 20, 100, 40, 5)
        with c2:
            tempo_max = st.slider("Tempo Máx. (min)", 1, 30, 8, 1)
        tortuosidade = st.slider("Fator de Tortuosidade", 1.0, 2.0, 1.3, 0.1)

    if "p-mediana" in modo:
        p_fixado = st.slider("Número de bases (p)", 1, 15, 3, 1)

# Presets rápidos (cartesiano), além do upload
if modo_cartesiano:
    preset_cart = st.selectbox(
        "Pontos pré-determinados (cartesiano)",
        ["(nenhum)", "Exemplo 1 (3 pontos)", "Exemplo 2 (4 pontos)", "Exemplo 3 (2 pontos)"],
    )
else:
    preset_cart = "(nenhum)"

# Upload (para AMBOS os modos)
uploaded_file = st.file_uploader("Carregar dados (CSV/Excel)", type=["csv", "xlsx"])

df_inicial = None

# Presets (somente no modo cartesiano)
# Presets (somente no modo cartesiano)
if modo_cartesiano and preset_cart != "(nenhum)":
    if preset_cart == "Exemplo 1 (3 pontos)":
        df_inicial = pd.DataFrame(
            {"bairro": ["A", "B", "C"], "x": [-7, 6, 2], "y": [5, -4, -8], "peso": [4, 6, 5]}
        )
    elif preset_cart == "Exemplo 2 (4 pontos)":
        df_inicial = pd.DataFrame(
            {"bairro": ["A", "B", "C", "D"], "x": [-8, -2, 4, 9], "y": [7, -6, 1, -3], "peso": [2, 5, 3, 4]}
        )
    else:
        df_inicial = pd.DataFrame(
            {"bairro": ["A", "B"], "x": [-6, 8], "y": [4, -5], "peso": [1, 1]}
        )

# Upload (agora FORA do bloco acima)
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            try:
                df_temp = pd.read_csv(uploaded_file, sep=";")
                if len(df_temp.columns) < 2:
                    uploaded_file.seek(0)
                    df_temp = pd.read_csv(uploaded_file, sep=",")
            except Exception:
                uploaded_file.seek(0)
                df_temp = pd.read_csv(uploaded_file, sep=None, engine="python")
        else:
            df_temp = pd.read_excel(uploaded_file)

        if modo_cartesiano:
            df_inicial = limpar_e_normalizar_dados_cartesiano(df_temp)
        else:
            df_inicial = limpar_e_normalizar_dados(df_temp)

        if df_inicial is not None and not df_inicial.empty:
            st.success(f"{len(df_inicial)} pontos carregados.")
        else:
            st.warning("Arquivo carregado, mas não encontrei pontos válidos.")
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")

# Defaults
if df_inicial is None or df_inicial.empty:
    if modo_cartesiano:
        df_inicial = pd.DataFrame(
            {
                "bairro": ["Ponto A", "Ponto B", "Ponto C"],
                "x": [-6.0, 7.0, 2.0],
                "y": [5.0, -4.0, 1.0],
                "peso": [4, 6, 5],
            }
        )
    else:
        df_inicial = pd.DataFrame(
            {
                "bairro": ["Farol", "Centro", "Trapiche"],
                "lat": [-9.6502, -9.6658, -9.6737],
                "lon": [-35.7352, -35.7350, -35.7486],
                "peso": [4, 6, 5],
            }
        )

# Editor conforme modo (com limpeza garantida antes do editor -> evita KeyError ['x','y'] / ['lat','lon'])
if modo_cartesiano:
    df_inicial = limpar_e_normalizar_dados_cartesiano(df_inicial)
    df_editado = st.data_editor(
        df_inicial[["bairro", "x", "y", "peso"]],
        num_rows="dynamic",
        height=260,
    )
    df_limpo = limpar_e_normalizar_dados_cartesiano(df_editado)
else:
    df_inicial = limpar_e_normalizar_dados(df_inicial)
    df_editado = st.data_editor(
        df_inicial[["bairro", "lat", "lon", "peso"]],
        num_rows="dynamic",
        height=260,
    )
    df_limpo = limpar_e_normalizar_dados(df_editado)

# Botão de cálculo (AGORA sim: para ambos os modos)
if st.button("🚀 Calcular Otimização", type="primary"):
    if df_limpo is None or len(df_limpo) < 1:
        st.error("Sem dados suficientes.")
    else:
        if modo == "📍 1-centro (Minimax)":
            df_utm, zn, zl = preparar_dados_para_modelo(df_limpo, modo_cartesiano)
            if df_utm is None or df_utm.empty:
                st.error("Falha ao preparar dados (verifique colunas e valores).")
            else:
                sol = one_center_L1_weighted(df_utm, mode=center_mode) if usar_pesos else one_center_L1_unweighted(df_utm)
                cx, cy, R = sol["x"], sol["y"], sol["R"]

                lat_c, lon_c = (cx, cy) if modo_cartesiano else safe_to_latlon(cx, cy, zn, zl)

                R_cobertura = calcular_raio_taxi(df_utm, cx, cy)

                st.session_state["resultado"] = {
                    "tipo": "1centro",
                    "utm_zone": (zn, zl),
                    "usar_pesos": usar_pesos,
                    "center_mode": center_mode if usar_pesos else "nao_ponderado",
                    "centro": {
                        "lat": float(lat_c),
                        "lon": float(lon_c),
                        "x_utm": float(cx),
                        "y_utm": float(cy),
                        "R_star": float(R),
                        "R_cobertura": float(R_cobertura),
                        "info": sol["info"],
                        "regiao_xy_utm": sol["regiao"],
                    },
                }

        elif modo == "📍 1-mediana (Weber)":
            df_utm, zn, zl = preparar_dados_para_modelo(df_limpo, modo_cartesiano)
            if df_utm is None or df_utm.empty:
                st.error("Falha ao preparar dados (verifique colunas e valores).")
            else:
                x_med, y_med = one_median_L1(df_utm, usar_pesos=usar_pesos)
                lat_m, lon_m = (x_med, y_med) if modo_cartesiano else safe_to_latlon(x_med, y_med, zn, zl)

                reg_weber_2 = retangulo_solucoes_weber_2pontos(df_utm) if len(df_utm) == 2 else None

                st.session_state["resultado"] = {
                    "tipo": "1mediana",
                    "utm_zone": (zn, zl),
                    "usar_pesos": usar_pesos,
                    "mediana": {
                        "lat": float(lat_m),
                        "lon": float(lon_m),
                        "x_utm": float(x_med),
                        "y_utm": float(y_med),
                        "regiao_weber2_xy_utm": reg_weber_2,
                        "custo_total": float(soma_distancias_taxi(df_utm, x_med, y_med, usar_pesos=usar_pesos)),
                    },
                }

        elif modo == "⚡ p-centro (múltiplas bases)":
            modo_pcentro = center_mode if usar_pesos else "nao_ponderado"
            res = otimizar_pcentro(
                df_limpo,
                velocidade_kmh=velocidade,
                tortuosidade=tortuosidade,
                tempo_max_min=tempo_max,
                center_mode=modo_pcentro,
                modo_cartesiano=modo_cartesiano,
            )
            if res:
                centros, df_classificado, (zn, zl), dist_lim = res
                st.session_state["resultado"] = {
                    "tipo": "pcentro",
                    "pontos": centros,
                    "dados_classificados": df_classificado,
                    "params": {"v": velocidade, "t": tempo_max, "tau": tortuosidade, "dist_lim": dist_lim},
                    "utm_zone": (zn, zl),
                    "usar_pesos": usar_pesos,
                    "center_mode": modo_pcentro,
                }
                st.success(f"Solução encontrada: {len(centros)} base(s).")
            else:
                st.error("Não foi possível cobrir todos os pontos com os parâmetros dados.")

        else:  # p-mediana
            p_fixado_local = min(int(p_fixado), len(df_limpo))

            # Correção conceitual: p-mediana com p=1 DEVE coincidir com a 1-mediana
            if p_fixado_local == 1:
                df_utm, zn, zl = preparar_dados_para_modelo(df_limpo, modo_cartesiano)
                if df_utm is None or df_utm.empty:
                    st.error("Falha ao preparar dados (verifique colunas e valores).")
                else:
                    x_med, y_med = one_median_L1(df_utm, usar_pesos=usar_pesos)
                    lat_m, lon_m = (x_med, y_med) if modo_cartesiano else safe_to_latlon(x_med, y_med, zn, zl)

                    reg_weber_2 = retangulo_solucoes_weber_2pontos(df_utm) if len(df_utm) == 2 else None

                    st.session_state["resultado"] = {
                        "tipo": "1mediana",  # <- IMPORTANTÍSSIMO para o desenho usar o mesmo caminho da 1-mediana
                        "utm_zone": (zn, zl),
                        "usar_pesos": usar_pesos,
                        "mediana": {
                            "lat": float(lat_m),
                            "lon": float(lon_m),
                            "x_utm": float(x_med),
                            "y_utm": float(y_med),
                            "regiao_weber2_xy_utm": reg_weber_2,
                            "custo_total": float(soma_distancias_taxi(df_utm, x_med, y_med, usar_pesos=usar_pesos)),
                        },
                    }
                    st.success("p-mediana com p=1 coincide com a 1-mediana (resultado exibido como 1-mediana).")

            else:
                res = resolver_pmediana(
                    df_limpo,
                    p=p_fixado_local,
                    usar_pesos=usar_pesos,
                    modo_cartesiano=modo_cartesiano,
                )
                if res:
                    centros, df_classificado, (zn, zl), custo_obj = res
                    st.session_state["resultado"] = {
                        "tipo": "pmediana",
                        "pontos": centros,
                        "dados_classificados": df_classificado,
                        "params": {"p": p_fixado_local, "custo_obj": float(custo_obj)},
                        "utm_zone": (zn, zl),
                        "usar_pesos": usar_pesos,
                    }
                    st.success(f"Solução p-mediana calculada com p={p_fixado_local}.")
                else:
                    st.error("Não foi possível calcular a p-mediana com os dados atuais.")




    # Mostrar resultado (painel)
    if "resultado" in st.session_state:
        res = st.session_state["resultado"]

        if res["tipo"] == "1centro":
            ct = res["centro"]
            st.markdown(
                "<div class='result-box' style='border-left:6px solid #1e40af;'>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<div style='margin-bottom:10px; font-size:13px; color:#475569;'>Modelo selecionado:</div>",
                unsafe_allow_html=True
            )

            st.write(f"**Usa pesos?** {'Sim' if res['usar_pesos'] else 'Não'}")
            st.write(f"**Modelo minimax**: `{res['center_mode']}`")
            st.write(f"**Raio de cobertura (L1)**: {ct['R_cobertura']:.1f}")
            st.write(f"**Tipo do conjunto ótimo**: `{ct['info'].get('tipo','—')}`")
            st.markdown("</div>", unsafe_allow_html=True)

        elif res["tipo"] == "1mediana":
            md = res["mediana"]
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.metric("📍 1-mediana (Weber, L1)", f"{md['lat']:.5f}, {md['lon']:.5f}")
            st.write(f"**Usa pesos?** {'Sim' if res['usar_pesos'] else 'Não'}")
            st.write(f"**Custo total (Σ distâncias L1)**: {md['custo_total']:.1f}")
            if md["regiao_weber2_xy_utm"] is not None:
                st.info("Caso com 2 pontos: o conjunto ótimo da 1-mediana é um retângulo em L1.")
            st.markdown("</div>", unsafe_allow_html=True)

        elif res["tipo"] == "pcentro":
            st.markdown("### 📊 Resultado p-centro")
            st.write(f"**Usa pesos?** {'Sim' if res['usar_pesos'] else 'Não'}")
            st.write(f"**Modelo minimax**: `{res['center_mode']}`")
            for i, pt in enumerate(res["pontos"]):
                st.markdown(
                    f"""
                    <div class="result-box" style="padding: 10px;">
                        <b>Base {i+1}</b> • Atende {pt['pontos_atendidos']} locais • raio(cluster)={pt['raio_cluster']:.0f}<br>
                        Coordenadas: {pt['lat']:.6f}, {pt['lon']:.6f}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:  # pmediana
            st.markdown("### 📊 Resultado p-mediana")
            st.write(f"**Usa pesos?** {'Sim' if res['usar_pesos'] else 'Não'}")
            st.write(f"**p** = {res['params']['p']} | **Custo objetivo total** = {res['params']['custo_obj']:.1f}")
            for i, pt in enumerate(res["pontos"]):
                st.markdown(
                    f"""
                    <div class="result-box" style="padding: 10px;">
                        <b>Base {i+1}</b> • Atende {pt['pontos_atendidos']} locais • custo(cluster)={pt['custo_cluster']:.1f}<br>
                        Coordenadas: {pt['lat']:.6f}, {pt['lon']:.6f}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)
    # =========================
    # MODO CARTESIANO
    # =========================
with col2:
    if modo_cartesiano:
        df_utm, _, _ = preparar_dados_para_modelo(df_limpo, True)

        if df_utm is None or df_utm.empty:
            st.warning("Sem pontos válidos no modo cartesiano (verifique colunas x/y).")
        else:
            # Faixa inicial para zoom
            xmin, xmax = float(df_utm["x_utm"].min()), float(df_utm["x_utm"].max())
            ymin, ymax = float(df_utm["y_utm"].min()), float(df_utm["y_utm"].max())
            dx = max(1.0, xmax - xmin)
            dy = max(1.0, ymax - ymin)

            pad = 0.25
            x0 = xmin - pad * dx
            x1 = xmax + pad * dx
            y0 = ymin - pad * dy
            y1 = ymax + pad * dy

            st.caption("Zoom/Janela de visualização (modo cartesiano)")
            cA, cB = st.columns(2)
            with cA:
                xlim = st.slider("Janela em X", min_value=float(x0 - 2*dx), max_value=float(x1 + 2*dx),
                                 value=(float(x0), float(x1)))
            with cB:
                ylim = st.slider("Janela em Y", min_value=float(y0 - 2*dy), max_value=float(y1 + 2*dy),
                                 value=(float(y0), float(y1)))

            fig, ax = plt.subplots(figsize=(8, 8))

# Título elegante
            ax.set_title(
                "Geometria Táxi (L1) — Sistema Cartesiano",
                fontsize=14,
                fontweight="bold",
                pad=15
            )

# Eixos mais discretos
            ax.set_xlabel("Eixo X", fontsize=11)
            ax.set_ylabel("Eixo Y", fontsize=11)

# Remover bordas superiores e direitas
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

# Suavizar grade
            ax.grid(True, linestyle="--", alpha=0.25)

# Fundo levemente claro
            ax.set_facecolor("#f9fbff")


            ax.scatter(
                df_utm["x_utm"],
                df_utm["y_utm"],
                s=70,
                alpha=0.85,
                edgecolors="white",
                linewidth=0.8
            )


            if "resultado" in st.session_state:
                res = st.session_state["resultado"]

                if res["tipo"] in {"pcentro", "pmediana"}:
                    for pt in res["pontos"]:
                        cx = pt["x_utm"]
                        cy = pt["y_utm"]
                        ax.scatter(
                            cx,
                            cy,
                            marker="X",
                            s=180,
                            edgecolors="black",
                            linewidth=1
                        )


                        r = pt.get("raio_cluster", 0.0)
                        losango = [(cx + r, cy), (cx, cy + r), (cx - r, cy), (cx, cy - r), (cx + r, cy)]
                        xs, ys = zip(*losango)
                        ax.plot(xs, ys, linewidth=2.2, alpha=0.9)


                elif res["tipo"] == "1centro":
                    ct = res["centro"]
                    cx = ct["x_utm"]
                    cy = ct["y_utm"]
                    r = ct.get("R_cobertura", 0.0)

                    ax.scatter(cx, cy, s=200)

                    # losango mínimo (raio)
                    losango = [(cx + r, cy), (cx, cy + r), (cx - r, cy), (cx, cy - r), (cx + r, cy)]
                    xs, ys = zip(*losango)
                    ax.plot(xs, ys)

                    # conjunto ótimo degenerado (segmento/ponto/região)
                    tipo = ct.get("info", {}).get("tipo", "ponto")
                    regiao = ct.get("regiao_xy_utm", None)
                    plot_conjunto_otimo(ax, regiao, tipo, linestyle="--")

                elif res["tipo"] == "1mediana":
                    md = res["mediana"]
                    mx = md["x_utm"]
                    my = md["y_utm"]
                    ax.scatter(mx, my, s=220, marker="X")

                    reg = md.get("regiao_weber2_xy_utm", None)
                    if reg is not None:
                        plot_conjunto_otimo(ax, reg, "regiao", linestyle="--")

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True)
            st.pyplot(fig)

    # =========================
    # MODO GEOGRÁFICO (FOLIUM)
    # =========================
    else:

        centro_lat = df_limpo["lat"].mean() if not df_limpo.empty else -9.66
        centro_lon = df_limpo["lon"].mean() if not df_limpo.empty else -35.74

        m = folium.Map(
            location=[centro_lat, centro_lon],
            zoom_start=13,
            tiles="CartoDB Positron"
        )


        cores = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

        if not df_limpo.empty:
            cluster_map = {}

            if (
                "resultado" in st.session_state
                and st.session_state["resultado"]["tipo"] in {"pcentro", "pmediana"}
            ):
                df_res = st.session_state["resultado"]["dados_classificados"]

                # Blindagem: se df_res não tiver lat/lon (ex.: resultado vindo do modo cartesiano),
                # não tenta construir cluster_map com colunas inexistentes.
                if {"lat", "lon", "cluster"}.issubset(df_res.columns):
                    for _, row in df_res.iterrows():
                        cluster_map[(float(row["lat"]), float(row["lon"]))] = int(row["cluster"])
                else:
                    # Opcional: avisar uma vez e não quebrar o app
                    st.warning("Resultado atual não possui lat/lon (provavelmente calculado no modo cartesiano). Recalcule no modo geográfico para ver clusters no mapa.")
                    cluster_map = {}


            for _, row in df_limpo.iterrows():

                cor_ponto = "#555"
                cid = cluster_map.get((float(row["lat"]), float(row["lon"])), None)

                if cid is not None:
                    cor_ponto = cores[cid % len(cores)]

                folium.CircleMarker(
                    [float(row["lat"]), float(row["lon"])],
                    radius=5,
                    color=cor_ponto,
                    fill=True,
                    fill_opacity=0.85,
                    tooltip=f"{row['bairro']} (peso={row['peso']})",
                ).add_to(m)

                if cid is not None and "resultado" in st.session_state:
                    base = st.session_state["resultado"]["pontos"][cid]
                    folium.PolyLine(
                        [
                            [float(row["lat"]), float(row["lon"])],
                            [base["lat"], base["lon"]],
                        ],
                        color=cor_ponto,
                        weight=1,
                        opacity=0.3,
                    ).add_to(m)
        st.markdown("---")
        st.markdown("## 📊 Resultado da Otimização")

        if "resultado" in st.session_state:

            res = st.session_state["resultado"]

            if res["tipo"] == "1centro":

                zn, zl = res["utm_zone"]
                mm = res["centro"]

                # Losango mínimo que cobre os pontos (raio de cobertura)
                folium.Polygon(
                    locations=losango_L1_latlon(
                        mm["x_utm"], mm["y_utm"], mm["R_cobertura"], zn, zl
                    ),
                    color="red",
                    weight=2,
                    fill=False,
                    tooltip="Bola L1 mínima que cobre os pontos (raio de cobertura)",
                ).add_to(m)

                # Conjunto ótimo do 1-centro (ponto/segmento/região)
                reg = mm.get("regiao_xy_utm", None)
                tipo = mm.get("info", {}).get("tipo", "ponto")

                if reg is not None and tipo == "segmento":
                    seg_latlon = utm_xy_to_latlon_list(reg, zn, zl)
                    folium.PolyLine(
                        locations=seg_latlon,
                        color="orange",
                        weight=5,
                        opacity=0.9,
                        tooltip="Conjunto ótimo do 1-centro (segmento)",
                    ).add_to(m)

                elif reg is not None and tipo == "regiao":
                    poly_latlon = utm_xy_to_latlon_list(reg, zn, zl)
                    if len(poly_latlon) > 0:
                        poly_latlon.append(poly_latlon[0])
                    folium.Polygon(
                        locations=poly_latlon,
                        color="orange",
                        weight=3,
                        fill=True,
                        fill_opacity=0.10,
                        tooltip="Conjunto ótimo do 1-centro (região)",
                    ).add_to(m)

                # Ponto retornado pelo solver (representante)
                folium.Marker(
                    [mm["lat"], mm["lon"]],
                    popup="1-centro (Minimax, L1)",
                    icon=folium.Icon(color="red"),
                ).add_to(m)


            elif res["tipo"] == "1mediana":

                zn, zl = res["utm_zone"]
                wb = res["mediana"]

                # Região ótima (quando existir: caso 2 pontos)
                reg2 = wb.get("regiao_weber2_xy_utm", None)
                if reg2 is not None:
                    rect_latlon = utm_xy_to_latlon_list(reg2, zn, zl)
                    if len(rect_latlon) > 0:
                        rect_latlon.append(rect_latlon[0])

                    folium.Polygon(
                        locations=rect_latlon,
                        color="green",
                        weight=3,
                        fill=True,
                        fill_opacity=0.10,
                        dash_array="5, 8",
                        tooltip="Conjunto ótimo da 1-mediana (caso 2 pontos)",
                    ).add_to(m)

                folium.Marker(
                    [wb["lat"], wb["lon"]],
                    popup="1-mediana (Weber, L1)",
                    icon=folium.Icon(color="green"),
                ).add_to(m)


            elif res["tipo"] == "pcentro":

                for i, pt in enumerate(res["pontos"]):
                    cor_base = cores[i % len(cores)]

                    folium.Marker(
                        [pt["lat"], pt["lon"]],
                        popup=f"Base p-centro {i+1}",
                        icon=folium.Icon(color="red"),
                    ).add_to(m)

                    if pt["cobertura_poly"] is not None:
                        folium.Polygon(
                            locations=pt["cobertura_poly"],
                            color=cor_base,
                            weight=2,
                            fill=False,
                        ).add_to(m)

            elif res["tipo"] == "pmediana":

                for i, pt in enumerate(res["pontos"]):

                    cor_base = cores[i % len(cores)]

                    folium.Marker(
                        [pt["lat"], pt["lon"]],
                        popup=f"Base p-mediana {i+1}",
                        icon=folium.Icon(color="green"),
                    ).add_to(m)

                    folium.Polygon(
                        locations=losango_L1_latlon(
                            pt["x_utm"],
                            pt["y_utm"],
                            pt["raio_cluster"],
                            res["utm_zone"][0],
                            res["utm_zone"][1],
                            raio_max_plot=500_000,
                        ),
                        color=cor_base,
                        weight=2,
                        fill=False,
                    ).add_to(m)

        st_folium(m, width="100%", height=700)

    st.markdown(
        """
<div class="legend-box">
<b>Legenda rápida do mapa</b><br>
• <span style="color:#555">●</span> Pontos de demanda (cinza) / coloridos por cluster nos modos p.<br>
• <span style="color:red">⛑</span> Marcador vermelho: base de 1-centro ou p-centro.<br>
• <span style="color:green">📍</span> Marcador verde: base de 1-mediana ou p-mediana.<br>
• Linha fina colorida: ligação ponto → base do seu cluster.<br>
• Polígono tracejado: limite de cobertura no p-centro.<br>
• Losangos coloridos no p-mediana: cobertura visual em geometria Táxi (L1).<br>
• Losango vermelho: menor bola L1 que cobre os pontos no 1-centro.<br>
• Polígono laranja: conjunto ótimo degenerado do 1-centro (quando existir).<br>
• Polígono verde tracejado: conjunto ótimo da 1-mediana para 2 pontos.
</div>
""",
        unsafe_allow_html=True,
    )