# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io, csv, re
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AHP App — Robusto", layout="wide")

# =========================================================
# Cabeçalho e tutorial
# =========================================================
st.title("📊 FatorAHP — Analisador de Fatores e Pesos com AHP")
st.markdown("""
### 📝 Como usar este aplicativo

1. **Carregue o arquivo CSV** contendo **apenas os fatores numéricos** que deseja analisar  
   - O arquivo deve conter:  
     - Primeira linha = nomes das variáveis (ex: `Fator1;Fator2;Fator3`).  
     - Demais linhas = valores numéricos.  
   - ⚠️ Não inclua colunas de identificação, nomes ou códigos — apenas os fatores.

2. **Estatísticas Descritivas**  
   - O app calcula média, desvio padrão, coeficiente de variação (CV), quartis, etc.

3. **Histogramas com densidade**  
   - Para verificar a distribuição de cada fator (simetria, outliers, modas).

4. **Matriz AHP**  
   - Use a sugestão automática (baseada em média + CV) ou preencha manualmente.  
   - No modo manual, edite apenas o triângulo superior — os inversos são preenchidos automaticamente.

5. **Cálculo final**  
   - O app exibe os pesos dos fatores, λmáx, CI e CR.  
   - Também mostra um ranking com classificação da importância (Muito Alta, Alta, Média, Baixa).

---
""")

# -------------------------
# Utilitários para leitura do CSV (robusto)
# -------------------------
def try_decode(raw_bytes):
    for enc in ('utf-8', 'latin1', 'cp1252'):
        try:
            s = raw_bytes.decode(enc)
            return s, enc
        except Exception:
            continue
    return raw_bytes.decode('utf-8', errors='replace'), 'utf-8'

def read_csv_flexible(uploaded_file):
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    text, encoding = try_decode(raw)

    # Tentativas comuns: (sep, decimal)
    attempts = [
        (';', ','),   # comum em BR: ; e decimal vírgula
        (',', '.'),   # csv padrão en/us
        ('\t', '.'),  # tsv
        ('|', '.'),
    ]

    for sep, decimal in attempts:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, decimal=decimal, engine='python')
            # se resultou mais de 1 coluna plausível -> sucesso
            if df.shape[1] > 1:
                return df, sep, decimal, encoding
            # se apenas uma coluna, pode ainda estar correto (um campo apenas)
        except Exception:
            pass

    # Tentar csv.Sniffer (pode falhar quando há vírgulas decimais)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=[',',';','\t','|'])
        sep = dialect.delimiter
        # tentar duas opções de decimal
        for decimal in [',', '.']:
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep, decimal=decimal, engine='python')
                if df.shape[1] > 1:
                    return df, sep, decimal, encoding
            except Exception:
                pass
    except Exception:
        pass

    # fallback: se é uma única coluna com delimitadores dentro das linhas, split manualmente
    lines = [ln for ln in text.splitlines() if ln.strip()!='']
    if len(lines) > 0:
        # detecta separador mais frequente nas linhas
        counts = {d: sum(ln.count(d) for ln in lines[:20]) for d in [';', ',', '\t', '|']}
        sep = max(counts, key=counts.get)
        # split
        rows = [re.split(r'[;,\t|]', ln) for ln in lines]
        header = rows[0]
        data = rows[1:]
        df = pd.DataFrame(data, columns=header)
        # assumimos decimal = ',' se sep == ';' (heurística)
        decimal = ',' if sep == ';' else '.'
        return df, sep, decimal, encoding

    # última tentativa: usar pandas autodetect
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, engine='python')
        return df, ',', '.', encoding
    except Exception as e:
        raise ValueError(f"Não foi possível ler o CSV automaticamente: {e}")

def clean_and_convert(df, decimal_hint):
    df = df.copy()
    # limpar nomes
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        s = df[c].astype(str).str.strip()
        s = s.str.replace(r'(^"|"$)', '', regex=True)  # remove aspas extremas
        s = s.str.replace(r'\s+', '', regex=True)      # remove espaços
        # se decimal_hint for ',' tratamos possíveis milhares com '.' e vírgula como decimal
        if decimal_hint == ',':
            # remover pontos que atuam como separador de milhares (heurística)
            # somente se existirem padrões como \d.\d{3}
            s = s.str.replace(r'\.(?=\d{3}(?:[\.\,]|$))', '', regex=True)
            s = s.str.replace(',', '.')
        else:
            # remove vírgulas como separador de milhares
            s = s.str.replace(',', '', regex=True)
        df[c] = pd.to_numeric(s, errors='coerce')
    # drop colunas completamente nulas
    df = df.loc[:, df.notna().any(axis=0)]
    return df

# -------------------------
# AHP: gerar matriz a partir da heurística (distância média a 1)
# -------------------------
def gerar_matriz_sugestao(df_norm):
    # df_norm: DataFrame numérico (de preferência normalizado média=1)
    distancias = df_norm.apply(lambda col: np.nanmean(np.abs(col - 1)), axis=0).to_dict()
    # ordenar por menor distância (maior prioridade)
    ordenado = sorted(distancias.items(), key=lambda x: x[1])
    variaveis = [v for v,_ in ordenado]
    n = len(variaveis)
    # scores inteiros: 1º -> n, 2º -> n-1, ..., último -> 1
    scores_int = {variaveis[i]: n - i for i in range(n)}
    # construir matriz (usar razão de scores e arredondar para inteiro Saaty)
    matriz = np.ones((n,n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            ratio = scores_int[variaveis[i]] / scores_int[variaveis[j]]
            ratio_r = int(round(ratio))
            ratio_r = min(max(ratio_r, 1), 9)
            matriz[i,j] = ratio_r
            matriz[j,i] = 1.0 / ratio_r
    df_scores = pd.DataFrame({
        "Variável": variaveis,
        "Distância Média a 1": [distancias[v] for v in variaveis],
        "Score (inteiro)": [scores_int[v] for v in variaveis]
    })
    matriz_df = pd.DataFrame(matriz, index=variaveis, columns=variaveis)
    return matriz_df, df_scores

# -------------------------
# AHP: cálculo de pesos e índices
# -------------------------
def calcular_ahp_indices(A):
    # A deve ser matriz quadrada, recíproca (ou quase)
    A = np.array(A, dtype=float)
    n = A.shape[0]
    # Normalização por coluna e média das linhas (método de média - fallback)
    with np.errstate(divide='ignore', invalid='ignore'):
        col_sums = A.sum(axis=0)
        matriz_norm = A / col_sums
        pesos_rowmean = np.nanmean(matriz_norm, axis=1)
    # Método de autovetor (mais preciso)
    try:
        autovalores, autovetores = np.linalg.eig(A)
        # escolher autovalor com maior parte real
        index_max = np.argmax(autovalores.real)
        lambda_max = autovalores[index_max].real
        vetor = autovetores[:, index_max].real
        pesos = np.abs(vetor)
        if pesos.sum() == 0:
            pesos = pesos_rowmean
        else:
            pesos = pesos / pesos.sum()
    except Exception:
        # fallback
        pesos = pesos_rowmean
        lambda_max = np.nan

    # CI e CR
    CI = (lambda_max - n) / (n - 1) if n > 1 and not np.isnan(lambda_max) else 0.0
    RI_table = {1:0.00,2:0.00,3:0.58,4:0.90,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.49}
    RI = RI_table.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0.0
    return pesos, lambda_max, CI, CR

# -------------------------
# Função: aplica reciprocidade/valida escala Saaty a partir da matriz editada
# -------------------------
def enforce_reciprocity_and_scale(df_matrix):
    M = df_matrix.copy().astype(float).values
    n = M.shape[0]
    # garantir diagonal 1
    for i in range(n):
        M[i,i] = 1.0
    # para cada par i<j, decidir valor principal e aplicar inverso
    for i in range(n):
        for j in range(i+1, n):
            aij = M[i,j]
            aji = M[j,i]
            # se aij é nan ou zero, mas aji existe -> usar aji
            if not np.isfinite(aij) or aij == 0:
                if np.isfinite(aji) and aji != 0:
                    val = float(aji)
                    # se usuário preencheu menor que 1 assume que foi inverso então invert
                    if val < 1:
                        val = 1.0 / val
                    # clamp 1..9
                    val = min(max(val, 1.0), 9.0)
                    M[i,j] = val
                    M[j,i] = 1.0 / val
                else:
                    # nenhum valor fornecido -> assume 1
                    M[i,j] = 1.0
                    M[j,i] = 1.0
            else:
                val = float(aij)
                # se o usuário digitar um valor <1 no triângulo superior, interpretamos como inverso
                if val < 1:
                    val = 1.0 / val
                val = min(max(val, 1.0), 9.0)
                M[i,j] = val
                M[j,i] = 1.0 / val
    return pd.DataFrame(M, index=df_matrix.index, columns=df_matrix.columns)

# -------------------------
# UI e fluxo principal
# -------------------------
st.sidebar.header("Upload & Config")
uploaded = st.sidebar.file_uploader("Carregar CSV (sep: ; ou , ou \\t)", type=['csv', 'txt'])

if uploaded is not None:
    try:
        df_raw, detected_sep, detected_decimal, encoding = read_csv_flexible(uploaded)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo automaticamente: {e}")
        st.stop()

    st.markdown("### 🗂️ Dados carregados (preview)")
    st.write(f"**Detectado:** separador = `{detected_sep}`, decimal_hint = `{detected_decimal}`, encoding = `{encoding}`")
    st.dataframe(df_raw.head(8))

    # conversão
    df_clean = clean_and_convert(df_raw, detected_decimal)
    if df_clean.shape[1] == 0:
        st.error("Não foram detectadas colunas numéricas após a conversão. Verifique o separador/decimal do arquivo.")
        st.stop()

    st.markdown("### 📈 Estatísticas Descritivas (após conversão para numérico)")
    st.write(df_clean.describe().T)

    # opção de normalizar (média = 1)
    big_means = (df_clean.mean().abs() > 10).any()
    if big_means:
        st.warning("Algumas variáveis têm média muito alta — normalizar para média = 1 é recomendado para a heurística.")
    normalizar = st.checkbox("Normalizar variáveis para média = 1 (recomendado)", value=True)

    if normalizar:
        means = df_clean.mean().replace(0, np.nan)
        df_norm = df_clean.divide(means, axis=1).fillna(0)
    else:
        df_norm = df_clean.copy()

    st.markdown("### 🔎 Preview (dados usados para calcular Distância Média a 1)")
    st.dataframe(df_norm.head(8))

    # ---------- NOVO: histogramas COM DENSIDADE (antes do Passo 3) ----------
    st.markdown("### 📊 Visualização de Histograma com Densidade")
    st.caption("O histograma ajuda a identificar assimetrias, múltiplas modas e outliers — barras azuis (frequência/densidade) e linha vermelha (densidade).")

    cols = st.columns(3)
    for i, col in enumerate(df_norm.columns):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        try:
            sns.histplot(df_norm[col].dropna(), bins=20, stat="density", color="skyblue", alpha=0.7, ax=ax)
        except Exception:
            ax.hist(df_norm[col].dropna(), bins=20, density=True, alpha=0.6)
        try:
            sns.kdeplot(df_norm[col].dropna(), color="red", ax=ax, linewidth=2)
        except Exception:
            pass
        ax.set_title(f"Histograma e Curva de Densidade de {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Densidade")
        with cols[i % 3]:
            st.pyplot(fig)
    # -----------------------------------------------------------------------

    # gerar sugestão
    matriz_sugerida, tabela_scores = gerar_matriz_sugestao(df_norm)
    st.markdown("### 🎯 Passo 3: Tabela de Pontuações (base da sugestão AHP)")
    st.dataframe(tabela_scores)

    # escolha entre sugestão editável ou preencher manualmente
    st.markdown("---")
    st.markdown("### 🧮 Passo 4: Matriz de Comparação Pareada (Sugestão ou Manual)")
    modo = st.radio("Modo de preenchimento:", ["Usar Sugestão Automática", "Preencher Manualmente"], horizontal=True)

    variaveis = matriz_sugerida.index.tolist()
    n = len(variaveis)

    if modo == "Usar Sugestão Automática":
        st.info("Matriz sugerida (você pode editar os valores). Preferencialmente edite somente o triângulo superior; o sistema ajustará os inversos ao calcular.")
        try:
            # nova API
            edited = None
            try:
                edited = st.data_editor(matriz_sugerida, num_rows="fixed", use_container_width=True)
            except Exception:
                edited = st.experimental_data_editor(matriz_sugerida, num_rows="fixed", use_container_width=True)
            matriz_editada = edited.copy()
        except Exception:
            st.warning("Editor interativo não disponível — exibindo apenas a sugestão.")
            st.dataframe(matriz_sugerida)
            matriz_editada = matriz_sugerida.copy()
    else:
        st.info("Preencha manualmente os valores **apenas no triângulo superior** (acima da diagonal). "
                "Os valores da parte inferior serão preenchidos automaticamente com os inversos.")

        matriz_manual = pd.DataFrame(np.eye(n), index=variaveis, columns=variaveis)

        try:
            edited = None
            try:
                edited = st.data_editor(matriz_manual, num_rows="fixed", use_container_width=True)
            except Exception:
                edited = st.experimental_data_editor(matriz_manual, num_rows="fixed", use_container_width=True)

            # aplicar reciprocidade logo após edição
            matriz_editada = enforce_reciprocity_and_scale(edited.copy())

            st.subheader("Matriz (com inversos aplicados automaticamente)")
            st.dataframe(matriz_editada.style.format("{:.4f}"))

        except Exception:
            st.warning("Editor interativo não disponível — exibindo matriz identidade.")
            st.dataframe(matriz_manual)
            matriz_editada = matriz_manual.copy()
    
    # guardamos matriz editada na sessão
    st.session_state['matriz_editada'] = matriz_editada

    st.markdown("---")
    st.markdown("### 📌 Passo 5: Cálculo e Resultados do AHP")

    if st.button("Calcular AHP"):
        try:
            matriz_ed = st.session_state.get('matriz_editada').copy()
            # converter tudo para numérico
            for c in matriz_ed.columns:
                matriz_ed[c] = pd.to_numeric(matriz_ed[c], errors='coerce')

            # aplicar reciprocidade e escala Saaty
            matriz_final = enforce_reciprocity_and_scale(matriz_ed)

            # calcular índices e pesos
            pesos, lambda_max, CI, CR = calcular_ahp_indices(matriz_final.values)
            pesos = np.array(pesos, dtype=float)
            if pesos.sum() == 0:
                st.error("Erro: pesos calculados como zeros. Verifique os valores da matriz.")
                st.stop()
            pesos_pct = (pesos / pesos.sum()) * 100

            # exibir matriz final
            st.subheader("🔢 Matriz Pareada Final (reciprocidade aplicada)")
            st.dataframe(matriz_final.style.format("{:.4f}"))

            # exibir pesos e ranking
            df_pesos = pd.DataFrame({
                "Variável": matriz_final.index,
                "Peso (valor)": pesos,
                "Importância (%)": pesos_pct
            }).sort_values("Importância (%)", ascending=False).reset_index(drop=True)
            df_pesos["Classificação"] = df_pesos["Importância (%)"].apply(
                lambda p: "🟢 Muito Alta" if p >= 30 else ("🟡 Alta" if p >= 20 else ("🔵 Média" if p >= 10 else "🔴 Baixa"))
            )

            st.subheader("📊 Pesos dos Critérios e Classificação")
            st.dataframe(df_pesos.style.format({"Importância (%)":"{:.2f}", "Peso (valor)":"{:.6f}"}))

            # exibir índices de consistência
            st.subheader("📏 Índices de Consistência")
            st.write(f"- λmáx (autovalor máximo): **{lambda_max:.6f}**")
            st.write(f"- CI (Índice de Consistência): **{CI:.6f}**")
            st.write(f"- CR (Razão de Consistência): **{CR:.6f}**")
            if CR < 0.1:
                st.success("✅ Matriz consistente — julgamentos são lógicos.")
            else:
                st.error("⚠️ Matriz inconsistente — revise seus julgamentos!")

            # botão de download
            csv_bytes = df_pesos.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Baixar Pesos (CSV)", data=csv_bytes, file_name="ahp_pesos.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Erro no cálculo do AHP: {e}")

else:
    st.info("Carregue um CSV no sidebar. Dica: arquivos gerados no Brasil frequentemente usam `;` e `,` como decimal.")
