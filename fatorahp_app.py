# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from io import StringIO

st.set_page_config(page_title="📊 FatorAHP — Analisador de Fatores com AHP", layout="wide")
sns.set_style("whitegrid")

# ---------------------------
# Funções utilitárias
# ---------------------------
def try_decode(raw_bytes):
    for enc in ("utf-8", "latin1", "cp1252", "iso-8859-1"):
        try:
            return raw_bytes.decode(enc), enc
        except Exception:
            continue
    return raw_bytes.decode("utf-8", errors="replace"), "utf-8"

def detect_sep_by_counts(text, candidates=[';', ',', '\t', '|'], n_lines=50):
    lines = [ln for ln in text.splitlines() if ln.strip()!=''][:n_lines]
    if not lines:
        return ';'
    stats = {}
    for s in candidates:
        counts = [len(ln.split(s)) for ln in lines]
        stats[s] = (np.median(counts), np.std(counts))
    best = max(stats.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[0]
    return best

def read_csv_flexible(uploaded_file):
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        text, enc_guess = try_decode(raw)
    else:
        text = raw
        enc_guess = 'utf-8'
    # try common BR
    try:
        df = pd.read_csv(StringIO(text), sep=';', decimal=',', engine='c')
        return df, ';', ',', enc_guess
    except Exception:
        pass
    sep_guess = detect_sep_by_counts(text)
    for dec in [',', '.']:
        try:
            df = pd.read_csv(StringIO(text), sep=sep_guess, decimal=dec, engine='c')
            return df, sep_guess, dec, enc_guess
        except Exception:
            continue
    try:
        df = pd.read_csv(StringIO(text), engine='python')
        return df, 'auto', 'auto', enc_guess
    except Exception as e:
        raise e

def clean_and_numeric(df, decimal_hint=','):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        s = df[c].astype(str).str.strip()
        s = s.str.replace(r'(^"|"$)', '', regex=True)
        s = s.str.replace(r'\s+', '', regex=True)
        if decimal_hint == ',':
            s = s.str.replace(r'\.(?=\d{3}(?:[\.|,]|$))', '', regex=True)
            s = s.str.replace(',', '.')
        else:
            s = s.str.replace(',', '', regex=True)
        df[c] = pd.to_numeric(s, errors='coerce')
    # drop fully NaN columns
    df = df.loc[:, df.notna().any(axis=0)]
    return df

def descriptive_stats(df):
    stats = pd.DataFrame({
        "Média": df.mean(),
        "Mediana": df.median(),
        "Desvio Padrão": df.std(),
        "CV": (df.std() / df.mean()).replace([np.inf, -np.inf], np.nan).fillna(0),
        "Mínimo": df.min(),
        "Máximo": df.max(),
        "Q1": df.quantile(0.25),
        "Q3": df.quantile(0.75)
    })
    return stats

def plot_histograms_brazil(df, cols_per_row=2, wait=0.12):
    cols = st.columns(cols_per_row)
    for i, col in enumerate(df.columns):
        fig, ax = plt.subplots(figsize=(5,3))
        sns.histplot(df[col].dropna(), bins=30, stat="density", color="skyblue", alpha=0.7, ax=ax, edgecolor='black', linewidth=0.3)
        try:
            sns.kdeplot(df[col].dropna(), color="red", ax=ax, linewidth=2)
        except Exception:
            pass
        ax.set_title(f"Histograma e Curva de Densidade de {col}")
        # format axis ticks labels as text with comma
        try:
            xt = ax.get_xticks()
            ax.set_xticklabels([f"{x:.2f}".replace(".", ",") for x in xt])
            yt = ax.get_yticks()
            ax.set_yticklabels([f"{y:.2f}".replace(".", ",") for y in yt])
        except Exception:
            pass
        with cols[i % cols_per_row]:
            st.pyplot(fig)
        plt.close(fig)
        time.sleep(wait)

def compute_scores(df_for_scores):
    dist_med = df_for_scores.apply(lambda col: np.nanmean(np.abs(col - 1.0)))
    cv = (df_for_scores.std() / df_for_scores.mean().replace(0, np.nan)).fillna(0)
    combined = (dist_med + cv) / 2.0
    order = np.argsort(combined.values)  # smaller is better
    n = len(combined)
    scores_int = np.zeros(n, dtype=int)
    for rank_pos, idx in enumerate(order):
        scores_int[idx] = n - rank_pos
    df_scores = pd.DataFrame({
        "Variável": df_for_scores.columns,
        "Distância Média a 1": dist_med.values,
        "CV": cv.reindex(df_for_scores.columns).values,
        "Score (inteiro)": scores_int
    })
    df_scores = df_scores.sort_values("Score (inteiro)", ascending=False).reset_index(drop=True)
    return df_scores

def map_ratio_to_saaty(ratio):
    try:
        r = float(ratio)
    except Exception:
        return 1
    if r <= 0:
        return 1
    if r < 1:
        r = 1.0 / r
    r = min(max(r, 1.0), 9.0)
    return int(round(r))

def build_suggested_matrix_from_scores(df_scores):
    ordered = df_scores["Variável"].tolist()
    scores = df_scores.set_index("Variável")["Score (inteiro)"].to_dict()
    n = len(ordered)
    A = np.ones((n,n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            si = scores[ordered[i]]
            sj = scores[ordered[j]]
            ratio = si / sj if sj != 0 else 9.0
            val = map_ratio_to_saaty(ratio)
            A[i,j] = val
            A[j,i] = 1.0 / val
    return pd.DataFrame(A, index=ordered, columns=ordered)

def enforce_reciprocity_and_scale(df_matrix):
    M = df_matrix.copy().astype(float)
    n = M.shape[0]
    for i in range(n):
        M.iloc[i,i] = 1.0
    for i in range(n):
        for j in range(i+1,n):
            aij = M.iloc[i,j]
            aji = M.iloc[j,i]
            if pd.notna(aij) and aij != 0:
                val = float(aij)
                if val < 1:
                    val = 1.0 / val
                val = min(max(val, 1.0), 9.0)
                M.iloc[i,j] = val
                M.iloc[j,i] = 1.0/val
            elif pd.notna(aji) and aji != 0:
                val = float(aji)
                if val < 1:
                    val = 1.0 / val
                val = min(max(val, 1.0), 9.0)
                M.iloc[i,j] = val
                M.iloc[j,i] = 1.0/val
            else:
                M.iloc[i,j] = 1.0
                M.iloc[j,i] = 1.0
    return M

def ahp_calculation(A):
    eigvals, eigvecs = np.linalg.eig(A)
    max_eigval = np.max(np.real(eigvals))
    max_index = np.argmax(np.real(eigvals))
    w = np.real(eigvecs[:, max_index])
    w = w / np.sum(w)
    n = A.shape[0]
    CI = (max_eigval - n) / (n - 1) if n > 1 else 0.0
    RI_dict = {1:0.0,2:0.0,3:0.58,4:0.90,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.49}
    RI = RI_dict.get(n, 1.49)
    CR = CI / RI if RI != 0 else 0.0
    return w, max_eigval, CI, CR

def classify_percent(p):
    if p >= 30:
        return "🟢 Muito Alta"
    elif p >= 20:
        return "🟡 Alta"
    elif p >= 10:
        return "🔵 Média"
    else:
        return "🔴 Baixa"

def format_num_br(x, fmt="{:.6f}"):
    try:
        return fmt.format(x).replace(".", ",")
    except Exception:
        return x

# ---------------------------
# Interface principal
# ---------------------------
st.title("📊 FatorAHP — Analisador de Fatores com AHP")
st.markdown("""
**Instruções rápidas:**  
- Carregue um CSV (recomendado: separador `;` e decimal `,`).  
- Seleccione as variáveis numéricas que deseja usar no AHP (aparecem deselecionadas por padrão).
""")
st.markdown("---")

uploaded = st.file_uploader("Escolha um arquivo CSV (recomendado sep=';', decimal=',')", type=["csv","txt"])
if not uploaded:
    st.info("Carregue um CSV para iniciar.")
    st.stop()

# leitura
try:
    df_raw, detected_sep, detected_dec, enc = read_csv_flexible(uploaded)
except Exception as e:
    st.error(f"Erro ao ler o CSV. Mensagem: {e}")
    st.stop()

st.success(f"Lido com sucesso (sep='{detected_sep}', decimal='{detected_dec}', encoding='{enc}').")

# Preview ajustado (expander)
st.subheader("📊 Visualizar Dados Carregados")
with st.expander("Prévia dos dados (primeiras 10 linhas)"):
    st.markdown(
        "<div style='color:#444;background:#f9f9f9;padding:8px;border-radius:6px;'>"
        "⚠️ Confira se o arquivo contém apenas as variáveis que você deseja analisar no AHP."
        "</div>",
        unsafe_allow_html=True
    )
    st.dataframe(df_raw.head(10))

# converter e detectar colunas numéricas
try:
    df_clean = clean_and_numeric(df_raw, decimal_hint=detected_dec if detected_dec else ',')
except Exception as e:
    st.error(f"Erro ao processar dados: {e}")
    st.stop()

if df_clean.shape[1] == 0:
    st.error("Nenhuma coluna numérica detectada após conversão.")
    st.stop()

num_cols = df_clean.columns.tolist()
non_num_cols = [c for c in df_raw.columns if c not in num_cols]
if non_num_cols:
    with st.expander("Colunas não-numéricas detectadas (serão ignoradas):"):
        st.write(non_num_cols)

# seleção de variáveis (aparecem deselecionadas por padrão)
st.header("Passo 1 — Seleção de Variáveis")
selected = st.multiselect("Escolha variáveis (aparecem desmarcadas):", options=num_cols, default=[])

if len(selected) == 0:
    st.warning("Selecione ao menos uma variável numérica para continuar.")
    st.stop()

# parâmetros
min_valid_percent = st.sidebar.slider("Min % colunas não-nulas por linha (aplica-se às variáveis selecionadas)", 50, 100, 100, 5)
normalize = st.sidebar.checkbox("Normalizar variáveis para média = 1 (recomendado)", value=True)
cols_per_row = st.sidebar.select_slider("Colunas por fila nos histogramas", options=[1,2,3], value=2)
hist_sleep = st.sidebar.slider("Pausa (s) entre histogramas)", 0.0, 0.5, 0.12, 0.02)

# aplicar seleção e validar linhas
df_selected = df_clean[selected].apply(pd.to_numeric, errors='coerce')
n_selected = len(selected)
min_required = int(np.ceil(n_selected * (min_valid_percent / 100.0)))
df_valid = df_selected.dropna(thresh=min_required, axis=0).copy()

if df_valid.shape[0] == 0:
    st.error("Nenhuma linha válida após aplicar critérios. Ajuste o slider ou selecione menos variáveis.")
    st.stop()

st.success(f"{n_selected} variáveis selecionadas; {df_valid.shape[0]} linhas válidas após critério ({min_valid_percent}%).")

# normalização opcional (média = 1)
if normalize:
    means = df_valid.mean().replace(0, np.nan)
    df_for_scores = df_valid.divide(means, axis=1).fillna(0)
else:
    df_for_scores = df_valid.copy()

# Passo 2: Estatísticas + histogramas
st.header("Passo 2 — Estatísticas Descritivas + Histogramas")
stats = descriptive_stats(df_valid)
stats_display = stats.round(6).astype(object)
for c in stats_display.columns:
    stats_display[c] = stats_display[c].apply(lambda x: format_num_br(x, "{:.6f}") if pd.notna(x) else "")
st.dataframe(stats_display)

st.info("Nota: CV indica dispersão — variáveis com CV alto são mais discriminatórias.")
st.subheader("Histogramas com Densidade")
plot_histograms_brazil(df_valid, cols_per_row=cols_per_row, wait=hist_sleep)

# Passo 3: tabela de pontuações
st.header("Passo 3 — Tabela de Pontuações (Distância média a 1 | CV | Score inteiro)")
score_table = compute_scores(df_for_scores)
score_table_display = score_table.copy()
score_table_display["Distância Média a 1"] = score_table_display["Distância Média a 1"].map(lambda x: format_num_br(x, "{:.6f}"))
score_table_display["CV"] = score_table_display["CV"].map(lambda x: format_num_br(x, "{:.6f}"))
st.dataframe(score_table_display)

# Passo 4: matriz sugestão & edição (robusto)
st.header("Passo 4 — Matriz de Comparação Pareada (Sugestão automática)")
A_suggest_df = build_suggested_matrix_from_scores(score_table)
st.subheader("Matriz Sugerida (ordenada por Score — maior → menor)")
st.dataframe(A_suggest_df.style.format("{:.4f}"))

# inicializa session_state
if "A_matrix" not in st.session_state or st.session_state.get("matrix_order") != list(A_suggest_df.index):
    st.session_state["A_matrix"] = A_suggest_df.copy()
st.session_state["matrix_order"] = list(A_suggest_df.index)

st.info("✏️ Você pode editar a matriz abaixo. Preencha apenas o triângulo superior; inversos serão calculados automaticamente.")

# function: robust matrix editor
def matrix_editor_robust(df_initial):
    n = df_initial.shape[0]
    # try new data_editor
    if hasattr(st, "data_editor"):
        try:
            edited = st.data_editor(df_initial, num_rows="fixed", use_container_width=True)
            return edited
        except Exception:
            pass
    # try experimental_data_editor
    if hasattr(st, "experimental_data_editor"):
        try:
            edited = st.experimental_data_editor(df_initial, num_rows="fixed", use_container_width=True)
            return edited
        except Exception:
            pass
    # fallback: manual inputs (only if n reasonable)
    if n > 12:
        st.warning("Edição interativa não disponível na sua versão do Streamlit para matrizes grandes. A matriz sugerida será usada como final. Para editar matrizes grandes, atualize o Streamlit (recomendado).")
        return df_initial.copy()
    st.warning("Editor visual não disponível — exibindo campos numéricos para editar o triângulo superior.")
    edited = df_initial.copy()
    # render manual inputs row by row
    for i in range(n):
        cols = st.columns(n+1)
        cols[0].markdown(f"**{edited.index[i]}**")
        for j in range(n):
            if i == j:
                cols[j+1].markdown("1")
                edited.iat[i,j] = 1.0
            elif i < j:
                default_val = float(df_initial.iat[i,j]) if pd.notna(df_initial.iat[i,j]) else 1.0
                val = cols[j+1].number_input(f"{edited.index[i]} vs {edited.columns[j]}", min_value=0.001, max_value=100.0, value=default_val, step=0.1, format="%.4f", key=f"m_{i}_{j}")
                edited.iat[i,j] = val
            else:
                # leave lower triangle blank for now
                cols[j+1].markdown("")
    return edited

edited = matrix_editor_robust(st.session_state["A_matrix"])

# apply reciprocity
try:
    edited_df = edited.copy()
    A_enforced = enforce_reciprocity_and_scale(edited_df)
    A_enforced = A_enforced.reindex(index=st.session_state["matrix_order"], columns=st.session_state["matrix_order"])
    st.session_state["A_matrix"] = A_enforced
    st.subheader("Matriz (com reciprocidade aplicada — ordem por Score mantida)")
    st.dataframe(A_enforced.style.format("{:.4f}"))
except Exception as e:
    st.error(f"Erro ao aplicar reciprocidade: {e}")
    st.dataframe(st.session_state["A_matrix"].style.format("{:.4f}"))

# Passo 5: cálculo AHP
st.header("Passo 5 — Cálculo e Resultados do AHP")
if st.button("Calcular AHP"):
    try:
        A_final = st.session_state["A_matrix"].copy().astype(float)
        A_final = enforce_reciprocity_and_scale(A_final)
        w, lam, CI, CR = ahp_calculation(A_final.values)
        weights_df = pd.DataFrame({
            "Variável": list(A_final.index),
            "Peso (valor)": w,
            "Importância (%)": (w * 100)
        }).sort_values("Peso (valor)", ascending=False).reset_index(drop=True)

        weights_df["Classificação"] = weights_df["Importância (%)"].apply(classify_percent)

        # exibir tabela de pesos (formatada BR)
        display_df = weights_df.copy()
        display_df_fmt = display_df.copy()
        display_df_fmt["Peso (valor)"] = display_df_fmt["Peso (valor)"].map(lambda x: format_num_br(x, "{:.6f}"))
        display_df_fmt["Importância (%)"] = display_df_fmt["Importância (%)"].map(lambda x: format_num_br(x, "{:.2f}"))
        st.subheader("📊 Pesos dos Critérios e Classificação")
        st.dataframe(display_df_fmt, use_container_width=True)

        # índices de consistência
        st.subheader("📐 Índices de Consistência")
        st.write(f"- λmáx (autovalor máximo): **{format_num_br(lam, '{:.6f}')}**")
        st.write(f"- CI (Índice de Consistência): **{format_num_br(CI, '{:.6f}')}**")
        st.write(f"- CR (Razão de Consistência): **{format_num_br(CR, '{:.6f}')}**")
        if CR < 0.1:
            st.success("✅ Matriz consistente — julgamentos são lógicos.")
        else:
            st.warning("⚠️ Matriz inconsistente — revise seus julgamentos!")

        # ranking e download
        ranking = display_df.copy()
        ranking["Peso (valor)"] = ranking["Peso (valor)"].map(lambda x: float(x))
        ranking["Importância (%)"] = ranking["Importância (%)"].map(lambda x: float(x))
        ranking = ranking.sort_values("Peso (valor)", ascending=False).reset_index(drop=True)

        st.subheader("🏆 Ranking Final")
        ranking_fmt = ranking.copy()
        ranking_fmt["Peso (valor)"] = ranking_fmt["Peso (valor)"].map(lambda x: format_num_br(x, "{:.6f}"))
        ranking_fmt["Importância (%)"] = ranking_fmt["Importância (%)"].map(lambda x: format_num_br(x, "{:.2f}"))
        st.dataframe(ranking_fmt, use_container_width=True)

        # export CSV com vírgula decimal
        out_df = ranking.copy()
        out_df["Peso (valor)"] = out_df["Peso (valor)"].map(lambda x: format_num_br(x, "{:.6f}"))
        out_df["Importância (%)"] = out_df["Importância (%)"].map(lambda x: format_num_br(x, "{:.6f}"))
        csv_main = out_df.to_csv(index=False, sep=';', encoding='utf-8-sig')

        extra = "\n\nÍndices de Consistência:\n"
        extra += f"λmáx;{format_num_br(lam,'{:.6f}')}\n"
        extra += f"CI;{format_num_br(CI,'{:.6f}')}\n"
        extra += f"CR;{format_num_br(CR,'{:.6f}')}\n"
        final_bytes = (csv_main + extra).encode('utf-8-sig')

        st.download_button("⬇️ Baixar ranking/pesos + índices (CSV)", data=final_bytes, file_name="ahp_ranking_indices.csv", mime="text/csv; charset=utf-8")
    except Exception as e:
        st.error(f"Erro no cálculo do AHP: {e}")

