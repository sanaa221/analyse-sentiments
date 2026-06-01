# ════════════════════════════════════════════════════════════
#   app.py  —  Darija Sentiment Analyzer  |  Interface Streamlit
#   Lancer avec :  streamlit run app.py
# ════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import seaborn as sns
from wordcloud import WordCloud
import re, joblib, os, time
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════
#  CONFIG PAGE
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Darija Sentiment Analyzer",
    page_icon="🇲🇦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════
#  CSS GLOBAL
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
  --bg:        #0D0F1A;
  --bg2:       #13162A;
  --bg3:       #1C2040;
  --accent:    #C8A96E;
  --accent2:   #E8C98E;
  --pos:       #4ADE80;
  --neg:       #F87171;
  --neu:       #60A5FA;
  --text:      #F0EDE8;
  --muted:     #8A8FA8;
  --border:    rgba(200,169,110,0.20);
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

/* ── Sidebar simulée ── */
.sidebar-sim {
  background: var(--bg2);
  border-right: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem;
  height: 100%;
}

/* ── Header ── */
.hero {
  background: linear-gradient(135deg, #1C2040 0%, #0D0F1A 60%, #1a1506 100%);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 2.2rem 2.5rem;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute; top: -60px; right: -60px;
  width: 220px; height: 220px;
  background: radial-gradient(circle, rgba(200,169,110,0.12) 0%, transparent 70%);
  border-radius: 50%;
}
.hero h1 {
  font-family: 'Syne', sans-serif;
  font-size: 2.4rem; font-weight: 800;
  color: var(--accent2) !important;
  margin: 0 0 .4rem;
  letter-spacing: -1px;
}
.hero p { color: var(--muted) !important; margin: 0; font-size: 1rem; }

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
  flex: 1;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.3rem 1.5rem;
  text-align: center;
  transition: transform .2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-card .val {
  font-family: 'Syne', sans-serif;
  font-size: 2.1rem; font-weight: 700;
  color: var(--accent2);
}
.metric-card .lbl { font-size: .78rem; color: var(--muted); margin-top: 4px; }

/* ── Section titles ── */
.section-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.15rem; font-weight: 700;
  color: var(--accent);
  border-left: 3px solid var(--accent);
  padding-left: .75rem;
  margin: 1.8rem 0 1rem;
}

/* ── Result box ── */
.result-box {
  border-radius: 16px;
  padding: 1.6rem 2rem;
  margin: 1rem 0;
  border: 1.5px solid;
  text-align: center;
}
.result-pos { background: rgba(74,222,128,.10); border-color: var(--pos); }
.result-neg { background: rgba(248,113,113,.10); border-color: var(--neg); }
.result-neu { background: rgba(96,165,250,.10);  border-color: var(--neu); }
.result-label {
  font-family: 'Syne', sans-serif;
  font-size: 1.8rem; font-weight: 800;
}
.result-conf { font-size: .9rem; color: var(--muted); margin-top: .3rem; }

/* ── Progress bars ── */
.prob-row { display: flex; align-items: center; gap: .8rem; margin: .5rem 0; font-size: .9rem; }
.prob-bar-bg { flex: 1; background: var(--bg3); border-radius: 8px; height: 10px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 8px; transition: width .6s ease; }

/* ── Table ── */
.styled-table {
  width: 100%; border-collapse: collapse; font-size: .88rem;
}
.styled-table th {
  background: var(--bg3); color: var(--accent);
  padding: 10px 14px; text-align: left;
  font-family: 'Syne', sans-serif; font-size: .8rem;
  text-transform: uppercase; letter-spacing: .06em;
}
.styled-table td { padding: 9px 14px; border-bottom: 1px solid var(--border); }
.styled-table tr:hover td { background: var(--bg2); }
.badge {
  display: inline-block; padding: 3px 10px; border-radius: 20px;
  font-size: .75rem; font-weight: 600;
}
.badge-pos { background: rgba(74,222,128,.18); color: var(--pos); }
.badge-neg { background: rgba(248,113,113,.18); color: var(--neg); }
.badge-neu { background: rgba(96,165,250,.18);  color: var(--neu); }

/* ── Inputs ── */
.stTextArea textarea, .stTextInput input {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stButton button {
  background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
  color: #0D0F1A !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 10px !important;
  padding: .55rem 1.4rem !important;
}
.stButton button:hover { opacity: .88; }

/* Selectbox, slider */
.stSelectbox > div, .stSlider > div { color: var(--text) !important; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  UTILS — PREPROCESSING
# ════════════════════════════════════════════════════════════
DARIJA_STOPWORDS = {
    'w','f','l','b','mn','li','had','hadi','dyal','ta3','wla','ila','lli',
    'wach','machi','kayen','kayna','dial','fih','fiha','men','wen','la','ma',
    'ch','shi','chi','hna','ghadi','kan','kant','walo','bla','kol','rah',
    'rani','bach','ida','3la','ela','7ta','9bal','ba3d','daba','mra','wqt',
    'wa9t','nhar','liyam','هاد','ديال','فيه','فيها','من','على','في','مع',
    'هو','هي','كان','كانت','لي','واش','ماشي','بلا','هنا','دابا','باش',
    'إلى','حتى','قبل','بعد','كل','راه'
}

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if t not in DARIJA_STOPWORDS and len(t) > 1]
    return ' '.join(tokens)

SENT_COLOR = {'positive': '#4ADE80', 'negative': '#F87171', 'neutral': '#60A5FA'}
SENT_EMOJI = {'positive': '😊', 'negative': '😠', 'neutral': '😐'}
SENT_LABEL_FR = {'positive': 'POSITIF', 'negative': 'NÉGATIF', 'neutral': 'NEUTRE'}


# ════════════════════════════════════════════════════════════
#  CHARGEMENT DES DONNÉES
# ════════════════════════════════════════════════════════════
@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path, encoding='latin-1')
    except:
        try:
            df = pd.read_csv(path, encoding='cp1256')
        except:
            df = pd.read_csv(path, encoding='utf-8')
    
    if 'text' not in df.columns:
        st.error("Le fichier CSV doit contenir une colonne 'text'")
        st.stop()
        
    if 'sentiment' not in df.columns:
        st.error("Le fichier CSV doit contenir une colonne 'sentiment'")
        st.stop()
    
    df['text_clean'] = df['text'].apply(preprocess)
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['text'].apply(lambda x: len(str(x)))
    return df

@st.cache_resource
def train_models(df):
    df = df.copy()
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['sentiment'])
    
    X = df['text_clean'].tolist()
    y = df['label'].tolist()
    
    X = np.array(X)
    y = np.array(y)
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25,
                                               random_state=42, stratify=y)
    
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000,
                            min_df=2, sublinear_tf=True)
    Xtr_v = tfidf.fit_transform(X_tr)
    Xte_v = tfidf.transform(X_te)

    trained = {}
    for name, clf in [
        ('Logistic Regression', LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ('Naive Bayes',         MultinomialNB(alpha=.5)),
        ('Linear SVM',          LinearSVC(C=1.0, max_iter=2000, random_state=42, dual='auto'))
    ]:
        clf.fit(Xtr_v, y_tr)
        y_pred = clf.predict(Xte_v)
        trained[name] = {
            'model': clf,
            'acc':   accuracy_score(y_te, y_pred),
            'f1':    f1_score(y_te, y_pred, average='weighted'),
            'cm':    confusion_matrix(y_te, y_pred),
            'report': classification_report(y_te, y_pred,
                                             target_names=le.classes_,
                                             output_dict=True)
        }
    return tfidf, le, trained, (X_te, y_te)


# ════════════════════════════════════════════════════════════
#  SIDEBAR SIMULÉE AVEC COLONNES
# ════════════════════════════════════════════════════════════

# Créer deux colonnes : une pour le menu (sidebar simulée) et une pour le contenu
col_sidebar, col_main = st.columns([1, 4])

with col_sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
      <div style='font-family:Syne,sans-serif; font-size:1.6rem;
                  font-weight:800; color:#C8A96E; letter-spacing:-1px;'>
        🇲🇦 Darija NLP
      </div>
      <div style='font-size:.75rem; color:#8A8FA8; margin-top:4px;'>
        Analyse des Sentiments
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠  Dashboard",
        "🔍  Prédiction",
        "📊  Modèles & Métriques",
        "📂  Données",
        "☁️  WordClouds",
        "ℹ️  À propos"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='font-size:.75rem; color:#8A8FA8;'>Dataset</div>",
                unsafe_allow_html=True)
    uploaded = st.file_uploader("Charger CSV", type=['csv'], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div class='section-title' style='font-size:.85rem;'>Modèle actif</div>",
                unsafe_allow_html=True)
    active_model = st.selectbox("", ['Logistic Regression','Naive Bayes','Linear SVM'],
                                 label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.72rem; color:#8A8FA8; line-height:1.6;'>
      Projet Deep Learning · Darija NLP<br>
      Analyse des sentiments · 2024
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  CHARGEMENT DES DONNÉES (suite)
# ════════════════════════════════════════════════════════════
DEFAULT_CSV = os.path.join("data", "darija_sentiment_dataset.csv")

if uploaded:
    df = load_data(uploaded)
elif os.path.exists(DEFAULT_CSV):
    df = load_data(DEFAULT_CSV)
else:
    with col_main:
        st.error(f"Erreur : Le fichier est introuvable à l'emplacement : {os.path.abspath(DEFAULT_CSV)}")
        st.info("Créez un dossier 'data' et placez-y le fichier 'darija_sentiment_dataset.csv'")
    st.stop()

# Entraînement des modèles
with st.spinner("Entraînement des modèles en cours..."):
    tfidf, le, trained, (X_te, y_te) = train_models(df)


# ════════════════════════════════════════════════════════════
#  PAGE : DASHBOARD
# ════════════════════════════════════════════════════════════
with col_main:
    if "Dashboard" in page:
        st.markdown("""
        <div class='hero'>
          <h1>Darija Sentiment Analyzer 🇲🇦</h1>
          <p>Analyse automatique des sentiments en langue arabe marocaine (Darija)</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Métriques ──
        total   = len(df)
        n_pos   = (df['sentiment'] == 'positive').sum()
        n_neg   = (df['sentiment'] == 'negative').sum()
        n_neu   = (df['sentiment'] == 'neutral').sum()

        best_f1 = max(v['f1'] for v in trained.values())

        st.markdown(f"""
        <div class='metric-row'>
          <div class='metric-card'>
            <div class='val'>{total}</div>
            <div class='lbl'>Exemples total</div>
          </div>
          <div class='metric-card'>
            <div class='val' style='color:#4ADE80'>{n_pos}</div>
            <div class='lbl'>😊 Positifs</div>
          </div>
          <div class='metric-card'>
            <div class='val' style='color:#F87171'>{n_neg}</div>
            <div class='lbl'>😠 Négatifs</div>
          </div>
          <div class='metric-card'>
            <div class='val' style='color:#60A5FA'>{n_neu}</div>
            <div class='lbl'>😐 Neutres</div>
          </div>
          <div class='metric-card'>
            <div class='val'>{best_f1:.2%}</div>
            <div class='lbl'>🏆 Meilleur F1</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='section-title'>Distribution des sentiments</div>",
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5), facecolor='#13162A')
            ax.set_facecolor('#13162A')
            vals   = [n_pos, n_neg, n_neu]
            labels = ['Positif', 'Négatif', 'Neutre']
            colors = ['#4ADE80', '#F87171', '#60A5FA']
            wedges, texts, autotexts = ax.pie(
                vals, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=90,
                wedgeprops=dict(linewidth=2, edgecolor='#0D0F1A')
            )
            for t in texts:    t.set_color('#F0EDE8')
            for a in autotexts: a.set_color('#0D0F1A'); a.set_fontweight('bold')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col2:
            st.markdown("<div class='section-title'>Sentiment × Source</div>",
                        unsafe_allow_html=True)
            cross = pd.crosstab(df['source'], df['sentiment'])
            fig, ax = plt.subplots(figsize=(6, 5), facecolor='#13162A')
            ax.set_facecolor('#13162A')
            cross.plot(kind='bar', ax=ax,
                       color=['#F87171','#60A5FA','#4ADE80'],
                       edgecolor='#0D0F1A', rot=35)
            ax.tick_params(colors='#8A8FA8', labelsize=8)
            ax.set_xlabel('', color='#8A8FA8')
            ax.set_ylabel('Nombre', color='#8A8FA8')
            ax.spines[:].set_color('#1C2040')
            ax.legend(facecolor='#1C2040', edgecolor='none',
                      labelcolor='#F0EDE8', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("<div class='section-title'>Distribution script</div>",
                        unsafe_allow_html=True)
            sc = df['script'].value_counts()
            fig, ax = plt.subplots(figsize=(5, 3), facecolor='#13162A')
            ax.set_facecolor('#13162A')
            ax.barh(sc.index, sc.values,
                    color=['#C8A96E', '#9B72CF'], edgecolor='#0D0F1A', height=.5)
            for i, v in enumerate(sc.values):
                ax.text(v + 2, i, str(v), va='center', color='#F0EDE8', fontsize=10)
            ax.tick_params(colors='#8A8FA8')
            ax.spines[:].set_color('#1C2040')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col4:
            st.markdown("<div class='section-title'>Longueur des textes</div>",
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3), facecolor='#13162A')
            ax.set_facecolor('#13162A')
            for s, c in zip(['positive','negative','neutral'],
                             ['#4ADE80','#F87171','#60A5FA']):
                df[df['sentiment']==s]['word_count'].hist(
                    ax=ax, bins=12, alpha=.65, label=s, color=c)
            ax.tick_params(colors='#8A8FA8', labelsize=8)
            ax.spines[:].set_color('#1C2040')
            ax.set_xlabel('Nombre de mots', color='#8A8FA8', fontsize=9)
            ax.legend(facecolor='#1C2040', edgecolor='none',
                      labelcolor='#F0EDE8', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # ── Exemples récents ──
        st.markdown("<div class='section-title'>Aperçu du dataset</div>",
                    unsafe_allow_html=True)
        sample = df[['text','sentiment','source','script']].sample(8, random_state=7)
        rows_html = ""
        for _, row in sample.iterrows():
            b = f"badge-{row['sentiment'][:3]}"
            label = SENT_LABEL_FR[row['sentiment']]
            rows_html += f"""
            <tr>
              <td>{row['text']}</td>
              <td><span class='badge {b}'>{label}</span></td>
              <td style='color:#8A8FA8'>{row['source']}</td>
              <td style='color:#8A8FA8'>{row['script']}</td>
            </tr>"""
        st.markdown(f"""
        <table class='styled-table'>
          <thead><tr><th>Texte</th><th>Sentiment</th><th>Source</th><th>Script</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE : PRÉDICTION
# ════════════════════════════════════════════════════════════
    elif "Prédiction" in page:
        st.markdown("<div class='hero'><h1>🔍 Prédiction en temps réel</h1><p>Analyse le sentiment d'une phrase en Darija</p></div>",
                    unsafe_allow_html=True)

        col_in, col_out = st.columns([1.1, 1])

        with col_in:
            st.markdown("<div class='section-title'>Entrez votre texte</div>",
                        unsafe_allow_html=True)
            user_text = st.text_area(
                "", height=140,
                placeholder="Ex: had lfilm zwiin bzzaf 3jbni mzzyan...\nأو: الماكلة كانت بنينة بزاف الله يبارك",
                label_visibility="collapsed"
            )

            examples = [
                "had lfilm zwiin bzzaf 3jbni mzzyan",
                "lkhadma khaiba bzzaf ma3jbatnich",
                "wach kayen chi jadid fhad lmaw9ou3",
                "الماكلة كانت بنينة بزاف الله يبارك",
                "هاد الخدمة خايبة ما عجباتنيش أبدا",
                "مازال ما عرفتش شنو دير"
            ]
            st.markdown("<div style='font-size:.8rem; color:#8A8FA8; margin-top:.5rem;'>Exemples rapides :</div>",
                        unsafe_allow_html=True)
            cols_ex = st.columns(2)
            for i, ex in enumerate(examples):
                if cols_ex[i % 2].button(f"'{ex[:28]}…'" if len(ex)>28 else f"'{ex}'",
                                          key=f"ex_{i}"):
                    user_text = ex

            predict_btn = st.button("⚡ Analyser le sentiment", use_container_width=True)

        with col_out:
            st.markdown("<div class='section-title'>Résultat</div>",
                        unsafe_allow_html=True)

            if predict_btn and user_text.strip():
                with st.spinner("Analyse en cours…"):
                    time.sleep(0.4)
                    clf   = trained[active_model]['model']
                    clean = preprocess(user_text)
                    vec   = tfidf.transform([clean])

                    if hasattr(clf, 'predict_proba'):
                        probs_arr = clf.predict_proba(vec)[0]
                    elif hasattr(clf, 'decision_function'):
                        raw = clf.decision_function(vec)[0]
                        exp_raw = np.exp(raw - np.max(raw))
                        probs_arr = exp_raw / exp_raw.sum()
                    else:
                        probs_arr = np.zeros(3)

                    pred_idx  = np.argmax(probs_arr)
                    pred_sent = le.inverse_transform([pred_idx])[0]
                    confidence = probs_arr[pred_idx]

                    css_cls = {'positive':'pos','negative':'neg','neutral':'neu'}[pred_sent]
                    col_var = {'positive':'--pos','negative':'--neg','neutral':'--neu'}[pred_sent]

                    st.markdown(f"""
                    <div class='result-box result-{css_cls}'>
                      <div style='font-size:2.5rem'>{SENT_EMOJI[pred_sent]}</div>
                      <div class='result-label' style='color:var({col_var})'>
                        {SENT_LABEL_FR[pred_sent]}
                      </div>
                      <div class='result-conf'>Confiance : {confidence:.1%} · {active_model}</div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
                    bar_colors = {'positive':'#4ADE80','negative':'#F87171','neutral':'#60A5FA'}
                    for i, cls in enumerate(le.classes_):
                        p = probs_arr[i]
                        w = int(p * 100)
                        st.markdown(f"""
                        <div class='prob-row'>
                          <div style='width:70px; text-align:right; color:#8A8FA8; font-size:.83rem;'>
                            {SENT_LABEL_FR[cls]}
                          </div>
                          <div class='prob-bar-bg'>
                            <div class='prob-bar-fill'
                                 style='width:{w}%; background:{bar_colors[cls]};'></div>
                          </div>
                          <div style='width:44px; color:#F0EDE8; font-size:.83rem;'>
                            {p:.1%}
                          </div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style='margin-top:1.2rem; padding:.9rem 1.2rem;
                                background:#1C2040; border-radius:10px;
                                font-size:.82rem; color:#8A8FA8;'>
                      <strong style='color:#C8A96E'>Texte préprocessé :</strong><br>{clean}
                    </div>""", unsafe_allow_html=True)

            elif predict_btn:
                st.warning("Veuillez entrer un texte.")
            else:
                st.markdown("""
                <div style='padding:3rem 1rem; text-align:center; color:#8A8FA8;'>
                  <div style='font-size:3rem'>🇲🇦</div>
                  <div style='margin-top:.7rem; font-size:.9rem;'>
                    Entrez une phrase en Darija<br>et cliquez sur Analyser
                  </div>
                </div>""", unsafe_allow_html=True)

        # ── Prédiction en lot ──
        st.markdown("---")
        st.markdown("<div class='section-title'>Prédiction en lot (batch)</div>",
                    unsafe_allow_html=True)
        batch_input = st.text_area(
            "Une phrase par ligne :", height=120,
            placeholder="had lfilm zwiin\nlkhadma khaiba\nwach kayen chi jadid",
            label_visibility="collapsed"
        )
        if st.button("Analyser tout le batch", use_container_width=False):
            lines = [l.strip() for l in batch_input.strip().split('\n') if l.strip()]
            if lines:
                clf = trained[active_model]['model']
                results_batch = []
                for line in lines:
                    clean = preprocess(line)
                    vec   = tfidf.transform([clean])
                    pred  = le.inverse_transform(clf.predict(vec))[0]
                    results_batch.append({'Texte': line,
                                           'Sentiment': SENT_LABEL_FR[pred],
                                           'Emoji': SENT_EMOJI[pred]})
                res_df = pd.DataFrame(results_batch)
                rows_html = ""
                for _, row in res_df.iterrows():
                    s_key = {v:k for k,v in SENT_LABEL_FR.items()}[row['Sentiment']]
                    b = f"badge-{s_key[:3]}"
                    rows_html += f"""
                    <tr>
                      <td>{row['Texte']}</td>
                      <td><span class='badge {b}'>{row['Emoji']} {row['Sentiment']}</span></td>
                    </tr>"""
                st.markdown(f"""
                <table class='styled-table'>
                  <thead><tr><th>Texte</th><th>Sentiment prédit</th></tr></thead>
                  <tbody>{rows_html}</tbody>
                </table>""", unsafe_allow_html=True)
                csv_dl = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Télécharger résultats CSV", csv_dl,
                                   "predictions.csv", "text/csv")


# ════════════════════════════════════════════════════════════
#  PAGE : MODÈLES & MÉTRIQUES
# ════════════════════════════════════════════════════════════
    elif "Modèles" in page:
        st.markdown("<div class='hero'><h1>📊 Modèles & Métriques</h1><p>Performance des modèles classiques TF-IDF</p></div>",
                    unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Comparaison des modèles</div>",
                    unsafe_allow_html=True)
        rows_html = ""
        best_f1 = max(v['f1'] for v in trained.values())
        for name, res in trained.items():
            crown = " 🏆" if res['f1'] == best_f1 else ""
            acc_bar = int(res['acc'] * 100)
            f1_bar  = int(res['f1']  * 100)
            rows_html += f"""
            <tr>
              <td style='font-weight:600'>{name}{crown}</td>
              <td>
                <div class='prob-row' style='margin:0'>
                  <div class='prob-bar-bg' style='flex:1'>
                    <div class='prob-bar-fill' style='width:{acc_bar}%; background:#C8A96E;'></div>
                  </div>
                  <div style='width:50px; font-size:.85rem'>{res['acc']:.4f}</div>
                </div>
              </td>
              <td>
                <div class='prob-row' style='margin:0'>
                  <div class='prob-bar-bg' style='flex:1'>
                    <div class='prob-bar-fill' style='width:{f1_bar}%; background:#9B72CF;'></div>
                  </div>
                  <div style='width:50px; font-size:.85rem'>{res['f1']:.4f}</div>
                </div>
              </td>
            </tr>"""
        st.markdown(f"""
        <table class='styled-table'>
          <thead><tr><th>Modèle</th><th>Accuracy</th><th>F1-score (weighted)</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Matrices de confusion</div>",
                    unsafe_allow_html=True)
        cols = st.columns(3)
        for col, (name, res) in zip(cols, trained.items()):
            with col:
                fig, ax = plt.subplots(figsize=(4, 3.5), facecolor='#13162A')
                ax.set_facecolor('#13162A')
                sns.heatmap(res['cm'], annot=True, fmt='d', cmap='YlOrBr',
                            ax=ax, xticklabels=le.classes_,
                            yticklabels=le.classes_,
                            linewidths=.5, linecolor='#0D0F1A',
                            annot_kws={'color':'#0D0F1A', 'fontweight':'bold'})
                ax.set_title(name, color='#C8A96E', fontsize=9, fontweight='bold')
                ax.set_xlabel('Prédit', color='#8A8FA8', fontsize=8)
                ax.set_ylabel('Réel',   color='#8A8FA8', fontsize=8)
                ax.tick_params(colors='#8A8FA8', labelsize=7)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

        st.markdown(f"<div class='section-title'>Rapport détaillé — {active_model}</div>",
                    unsafe_allow_html=True)
        report = trained[active_model]['report']
        rows_r = ""
        for cls in le.classes_:
            r = report[cls]
            rows_r += f"""
            <tr>
              <td><span class='badge badge-{cls[:3]}'>{SENT_LABEL_FR[cls]}</span></td>
              <td>{r['precision']:.3f}</td>
              <td>{r['recall']:.3f}</td>
              <td>{r['f1-score']:.3f}</td>
              <td>{int(r['support'])}</td>
            </tr>"""
        avg = report['weighted avg']
        rows_r += f"""<tr style='border-top:2px solid #C8A96E;'>
          <td style='color:#C8A96E; font-weight:600'>Weighted Avg</td>
          <td>{avg['precision']:.3f}</td>
          <td>{avg['recall']:.3f}</td>
          <td>{avg['f1-score']:.3f}</td>
          <td>{int(avg['support'])}</td>
        </tr>"""
        st.markdown(f"""
        <table class='styled-table'>
          <thead><tr><th>Classe</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr></thead>
          <tbody>{rows_r}</tbody>
        <table>""", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Accuracy vs F1 — tous modèles</div>",
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4), facecolor='#13162A')
        ax.set_facecolor('#13162A')
        names_m = list(trained.keys())
        accs_m  = [trained[m]['acc'] for m in names_m]
        f1s_m   = [trained[m]['f1']  for m in names_m]
        x_m = np.arange(len(names_m))
        b1 = ax.bar(x_m-.18, accs_m, .35, label='Accuracy', color='#C8A96E',  edgecolor='#0D0F1A')
        b2 = ax.bar(x_m+.18, f1s_m,  .35, label='F1 Weighted', color='#9B72CF', edgecolor='#0D0F1A')
        for b in list(b1)+list(b2):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+.005,
                    f'{b.get_height():.3f}', ha='center', va='bottom',
                    fontsize=8, color='#F0EDE8')
        ax.set_xticks(x_m)
        ax.set_xticklabels(names_m, color='#8A8FA8', fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.tick_params(colors='#8A8FA8')
        ax.spines[:].set_color('#1C2040')
        ax.legend(facecolor='#1C2040', edgecolor='none', labelcolor='#F0EDE8')
        ax.axhline(.8, color='#8A8FA8', ls='--', alpha=.4, lw=1)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ════════════════════════════════════════════════════════════
#  PAGE : DONNÉES
# ════════════════════════════════════════════════════════════
    elif "Données" in page:
        st.markdown("<div class='hero'><h1>📂 Exploration des données</h1><p>Parcourir et filtrer le dataset Darija</p></div>",
                    unsafe_allow_html=True)

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            filtre_sent = st.multiselect("Sentiment",
                ['positive','negative','neutral'],
                default=['positive','negative','neutral'])
        with col_f2:
            filtre_src  = st.multiselect("Source",
                df['source'].unique().tolist(),
                default=df['source'].unique().tolist())
        with col_f3:
            filtre_scr  = st.multiselect("Script",
                df['script'].unique().tolist(),
                default=df['script'].unique().tolist())

        filtered = df[
            df['sentiment'].isin(filtre_sent) &
            df['source'].isin(filtre_src) &
            df['script'].isin(filtre_scr)
        ]

        st.markdown(f"<div style='color:#8A8FA8; font-size:.85rem; margin:.5rem 0;'>"
                    f"{len(filtered)} exemples affichés sur {len(df)}</div>",
                    unsafe_allow_html=True)

        n_show = st.slider("Nombre de lignes", 10, min(200, len(filtered)), 30)
        sample_f = filtered.sample(min(n_show, len(filtered)), random_state=1)

        rows_html = ""
        for idx, row in sample_f.iterrows():
            b = f"badge-{row['sentiment'][:3]}"
            label = SENT_LABEL_FR[row['sentiment']]
            rows_html += f"""
            <tr>
              <td style='color:#8A8FA8; font-size:.8rem'>{idx+1}</td>
              <td>{row['text']}</td>
              <td><span class='badge {b}'>{label}</span></td>
              <td style='color:#8A8FA8'>{row['source']}</td>
              <td style='color:#8A8FA8'>{row['script']}</td>
              <td style='color:#8A8FA8'>{row['word_count']}</td>
            </tr>"""
        st.markdown(f"""
        <table class='styled-table'>
          <thead><tr><th>ID</th><th>Texte</th><th>Sentiment</th>
                     <th>Source</th><th>Script</th><th>Mots</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        csv_dl = filtered.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Télécharger les données filtrées",
                           csv_dl, "darija_filtered.csv", "text/csv")


# ════════════════════════════════════════════════════════════
#  PAGE : WORDCLOUDS
# ════════════════════════════════════════════════════════════
    elif "WordClouds" in page:
        st.markdown("<div class='hero'><h1>☁️ WordClouds</h1><p>Mots les plus fréquents par sentiment</p></div>",
                    unsafe_allow_html=True)

        script_wc = st.radio("Script à visualiser",
                              ['Tous','Latin seulement','Arabe seulement'],
                              horizontal=True)

        df_wc = df.copy()
        if script_wc == 'Latin seulement':  df_wc = df_wc[df_wc['script'] == 'latin']
        if script_wc == 'Arabe seulement':  df_wc = df_wc[df_wc['script'] == 'arabic']

        cmaps = {'positive':'Greens', 'negative':'Reds', 'neutral':'Blues'}
        titles = {'positive':'😊 POSITIF', 'negative':'😠 NÉGATIF', 'neutral':'😐 NEUTRE'}

        cols_wc = st.columns(3)
        for col, sent in zip(cols_wc, ['positive','negative','neutral']):
            with col:
                st.markdown(f"<div class='section-title'>{titles[sent]}</div>",
                            unsafe_allow_html=True)
                corpus = ' '.join(df_wc[df_wc['sentiment'] == sent]['text_clean'])
                if corpus.strip():
                    wc = WordCloud(
                        width=500, height=320,
                        background_color='#13162A',
                        colormap=cmaps[sent],
                        max_words=60, regexp=r"[\w']+",
                        collocations=False
                    ).generate(corpus)
                    fig, ax = plt.subplots(figsize=(4, 2.7), facecolor='#13162A')
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    plt.tight_layout(pad=0)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
                else:
                    st.info("Pas assez de données pour ce filtre.")

        st.markdown("<div class='section-title'>Top 15 mots par sentiment</div>",
                    unsafe_allow_html=True)
        cols_top = st.columns(3)
        for col, sent in zip(cols_top, ['positive','negative','neutral']):
            with col:
                corpus = ' '.join(df_wc[df_wc['sentiment'] == sent]['text_clean'])
                word_freq = Counter(corpus.split()).most_common(15)
                if word_freq:
                    words_t, freqs_t = zip(*word_freq)
                    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#13162A')
                    ax.set_facecolor('#13162A')
                    color = {'positive':'#4ADE80','negative':'#F87171','neutral':'#60A5FA'}[sent]
                    ax.barh(range(len(words_t)), freqs_t,
                            color=color, alpha=.8, edgecolor='#0D0F1A')
                    ax.set_yticks(range(len(words_t)))
                    ax.set_yticklabels(words_t, color='#F0EDE8', fontsize=8)
                    ax.tick_params(axis='x', colors='#8A8FA8', labelsize=7)
                    ax.spines[:].set_color('#1C2040')
                    ax.set_title(SENT_LABEL_FR[sent], color=color,
                                 fontfamily='monospace', fontsize=9)
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()


# ════════════════════════════════════════════════════════════
#  PAGE : À PROPOS
# ════════════════════════════════════════════════════════════
    elif "À propos" in page:
        st.markdown("<div class='hero'><h1>ℹ️ À propos du projet</h1><p>Analyse des sentiments en Darija marocaine</p></div>",
                    unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class='section-title'>Objectif</div>
            <p style='color:#8A8FA8; line-height:1.8; font-size:.93rem;'>
            Ce projet vise à construire un système d'<strong style='color:#C8A96E'>analyse automatique
            des sentiments</strong> pour la langue arabe marocaine (Darija), en utilisant
            des techniques de NLP et de Machine Learning.
            </p>

            <div class='section-title'>Technologies utilisées</div>
            """, unsafe_allow_html=True)

            techs = [
                ("🐍 Python", "Langage principal"),
                ("📒 Jupyter Notebook", "Environnement de développement"),
                ("🤗 Hugging Face", "Modèles pré-entraînés BERT"),
                ("⚙️ Scikit-learn", "Modèles classiques ML"),
                ("📊 TF-IDF", "Vectorisation du texte"),
                ("🔥 PyTorch", "Deep Learning"),
                ("📈 Matplotlib / Seaborn", "Visualisation"),
                ("🌐 Streamlit", "Interface web"),
            ]
            rows_tech = ""
            for name, desc in techs:
                rows_tech += f"""
                <tr>
                  <td style='color:#C8A96E; font-weight:600'>{name}</td>
                  <td style='color:#8A8FA8'>{desc}</td>
                </tr>"""
            st.markdown(f"""
            <table class='styled-table'>
              <thead><tr><th>Outil</th><th>Usage</th></tr></thead>
              <tbody>{rows_tech}</tbody>
            </table>""", unsafe_allow_html=True)

        with col_b:
            st.markdown("""
            <div class='section-title'>Pipeline NLP</div>
            """, unsafe_allow_html=True)

            steps = [
                ("1", "Collecte", "Dataset de phrases en Darija (Latin & Arabe)"),
                ("2", "Preprocessing", "Nettoyage, stopwords, normalisation"),
                ("3", "TF-IDF", "Vectorisation n-grammes (1,2)"),
                ("4", "Modèles classiques", "Logistic Regression, Naive Bayes, SVM"),
                ("5", "BERT", "CAMeLBERT fine-tuné sur dialecte arabe"),
                ("6", "Évaluation", "Accuracy, F1-score, matrice de confusion"),
                ("7", "Interface", "Streamlit dashboard interactif"),
            ]
            for num, title, desc in steps:
                st.markdown(f"""
                <div style='display:flex; gap:1rem; align-items:flex-start;
                            margin-bottom:.8rem; padding:.8rem 1rem;
                            background:#1C2040; border-radius:10px;
                            border-left:3px solid #C8A96E;'>
                  <div style='font-family:Syne,sans-serif; font-size:1.1rem;
                              font-weight:800; color:#C8A96E; min-width:24px;'>
                    {num}
                  </div>
                  <div>
                    <div style='font-weight:600; color:#F0EDE8; font-size:.9rem;'>{title}</div>
                    <div style='color:#8A8FA8; font-size:.82rem; margin-top:2px;'>{desc}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class='section-title'>Dataset</div>
            <div style='padding:.9rem 1.2rem; background:#1C2040; border-radius:10px;
                        font-size:.85rem; color:#8A8FA8; line-height:1.8;'>
              📌 <strong style='color:#C8A96E'>500 phrases</strong> annotées manuellement<br>
              🏷️ 3 classes : <span style='color:#4ADE80'>Positif</span> ·
                              <span style='color:#F87171'>Négatif</span> ·
                              <span style='color:#60A5FA'>Neutre</span><br>
              📝 Scripts : Franco-arabe (Latin) & Arabe<br>
              🌐 Sources : Facebook, Twitter, Google Reviews, Jumia, Avito…
            </div>""", unsafe_allow_html=True)