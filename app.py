import streamlit as st
import pandas as pd
import requests
import io
from transformers import AutoTokenizer
import time
import re
import plotly.express as px
from collections import Counter

# Jalan ini di terminal 2 : streamlit run app.py 
st.set_page_config(page_title="Astra Honda Sentiment Analysis", layout="wide")
api_url = "http://127.0.0.1:8000/predict" 

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

@st.cache_data
def load_kamus():
    try:
        # Pastikan file excel ada satu folder dengan ui.py
        kamus_data = pd.read_excel('kamuskatabaku.xlsx')
        return dict(zip(kamus_data['tidak_baku'], kamus_data['kata_baku']))
    except:
        return {}

tokenizer = load_tokenizer()
kamus_tidak_baku_dict = load_kamus()

custom_topic = {
            "Performance": [
                "tenaga", "power", "torsi", "torque", "speed", "kecepatan", "kencang", "ngebut", 
                "tarikan", "akselerasi", "napas", "top speed", "responsif", "jambak", "ngacir", 
                "lemot", "boyo", "berat", "ngempos", "gerung", "loyo", "kurang tenaga", 
                "nanjak", "kuat nanjak", "napas panjang", "napas pendek", "limiter", "rpm",
                "matic", "125", "150", "160", "sped", "tenagae", "tenagga", "powerr", "torq", "tork", "tros", "speeed",
                "kenceng", "kencang", "ngibrit", "ngeloyor", "ngeden", "akselarasi", "akselerasi",
                "nafas", "nafas panjang", "nafas pendek", "rpm tinggi", "rpm rendah", "napas", "mogok", "mati", "susah hidup",
                "brebet", "nyendat", "gredek", "panas", "overheat", "ngebul", "asap",
                "lemoot", "lelet", "beraaat", "narik", "tarikannya",
            ],

            # 2. DESIGN & ESTETIKA (Tampilan, Warna, Gaya)
            "Design": [
                "desain", "design", "designya", "model", "tampang", "bentuk", "gaya", "style", "look", "modern",
                "ganteng", "keren", "jelek", "aneh", "wagu", "kaku", "culun", "futuristik", 
                "retro", "klasik", "sporty", "elegan", "mewah", "garang", "agresif", "lancip",
                "headlamp", "lampu depan", "stoplamp", "buntut", "fairing", "striping", "stiker", 
                "livery", "decal", "facelift", "proporsional", "bongsor", "gambot", "ramping", "desainnya", "designnya", "modelnya",
                "gantengg", "kereen", "jelekk", "sporti", "elegant", "futuristik", "body", "bodynya",
                "lampu", "lamps","bongsorr", "rampingg", "modif", "stang"
            ],

            # 3. WARNA (Sub-kategori Design yang sering dibahas spesifik)
            "Color": [
                "warna", "cat", "paint", "doff", "matte", "glossy", "metalik", 
                "merah", "hitam", "putih", "biru", "silver", "abu", "kuning", "hijau",
                "two tone", "polos", "kelir", "pudar", "kusam", "belang", "krem" ,"warnanya", "colour",
                "item", "putihh", "abu2", "abu abu", "biruu", "silverr", "dof", "dop", "metalic",
                "two-tone", "twotone"
            ],

            # 4. BUILD QUALITY (Kualitas Material & Ketahanan)
            "Build Quality": [
                "build quality", "kualitas", "material", "bahan", "plastik", "besi", "baja",
                "rangka", "frame", "esaf", "keropos", "karat", "karatan", "patah", "lipat",
                "sambungan", "las", "las-lasan", "rapi", "kasar", "finishing", "celah", "gap",
                "bodi", "body", "getar", "koplak", "bunyi", "ringkih", "tipis", "tebal", "spakbor",
                "kokoh", "padat", "kopong", "awet", "tahan lama", "reog", "oblak",  "build", "kualitaas",
                "plastikan", "besinya", "geterr", "vibrasi", "ringkihh", "kerasa murahan", "kokohh", "padett",
                "reog", "oblag", "oblak", "tangguh", "solid", "berat", "kuat", "bobot", "tipis", "tipisin", "part", "rem", "remnya",
                "crancase", "pengereman", "bocor", "rembes", "netes", "oli", "getar",  "bunyi", "brisik", "klotok", "kasar"
            ],

            # 5. KENYAMANAN & ERGONOMI (Posisi Berkendara)
            "Comfort": [
                "nyaman", "enak", "keras", "empuk","ringan", "suspensi", "shock", "shockbreaker", 
                "jok", "busa", "kulit jok", "posisi duduk", "riding position", "segitiga", 
                "stang", "setang", "tekuk", "pegal", "pegel", "jinjit", "tinggi", "pendek", 
                "handling", "manurver", "lincah", "stabil", "anteng", "limbung", "oleng", 
                "boncengan", "pijakan", "dek", "luas", "sempit", "getaran mesin", "ground clearance",
                "polisi tidur","nyamann", "enaaak", "kerass", "empukkk", "handlingnya", "handlng",
                "lincahh", "antengg", "olengg", "limbungg", "bonceng", "dibonceng", "joknya", "suspensinya",
                "ban", "ban depan", "ban belakang", "licin", "grip", "nempel", "keras", "empuk",
                "stabil", "selip"
            ],

            # 6. FITUR & TEKNOLOGI
            "Features": [
                "fitur", "teknologi", "canggih", "lengkap", "minim fitur",
                "keyless", "smart key", "remote", "kunci", "kontak", "alarm", "answer back",
                "speedometer", "panel", "instrumen", "digital", "analog", "layar", "indikator",
                "abs", "cbs", "rem", "iss", "idling stop", "sss", "charger", "usb", "soket", 
                "lampu", "led", "projie", "drp", "hazard", "connected", "bluetooth", "fiturnya", "fitur2",
                "keyles", "key less", "spidometer", "spdometer", "speedo", "panelnya", "lednya", "usb charger",
                "bluetooh", "bluetoothnya", "error", "ga berfungsi", "mati", "delay"
            ],

            # 7. UTILITAS (Kepraktisan Harian)
            "Utility": [
                "bagasi", "penyimpanan", "laci", "konsol", "helm in", "muat helm", "helm",
                "tangki", "kapasitas", "liter", "galon", "dek rata", "gantungan", "hook",  "bagasinya", "bagasi luas",
                "muatt", "muat banyak", "helm fullface", "gantungan barang"
            ],

            # 8. EFISIENSI & BIAYA (Irit, Harga, Jual Kembali)
            "Fuel": [
                    "irit", "irittt", "boros", "boross", "hemat", "konsumsi", "konsumsi bbm",
                    "bbm", "bbmm", "bensin", "pertalite", "pertamax", "shell", "vpower",
                    "km/l", "kmpl", "km / l", "kml", "satu liter", "per liter",
                    "jarak tempuh", "isi full", "tangki", "kapasitas bensin"
            ],

            "Price": [
                    "harga", "harganya", "price", "mahal", "murah", "murahan",
                    "kemahalan", "overprice", "overpriced", "terjangkau", "worth it", "worthed", "worthit",
                    "jt", "jtaan", "juta", "jutaan", "cicilan", "kredit", "dp", "angsuran", "jutaaa",
                    "diskon", "promo", "resale", "resale value", "jual kembali", "second", "bekas", "anjlok", "umr"
            ],

            # 9. MASALAH TEKNIS (Troubleshooting & Keluhan Umum)
            "Issues": [
                "masalah", "penyakit", "minus", "keluhan", "cacat", "defect",
                "mogok", "mati", "susah hidup", "brebet", "nyendat", "gredek", "cv t getar",
                "bocor", "rembes", "netes", "oli", "asap", "ngebul", "vampir oli",
                "panas", "overheat", "kipas nyala", "radiator", 
                "bunyi", "brisik", "klotok", "nglitik", "kasar", "cit cit"
            ],

            # 10. AFTERSALES (Bengkel & Sparepart)
            "Aftersales": [
                "servis", "service", "bengkel", "mekanik", "montir", "antri", 
                "sparepart", "suku cadang", "onderdil", "inden", "langka", "gampang", "mudah",
                "ori", "kw", "aftermarket", "garansi", "klaim", "dealer", "sales"
            ]
        }

STOPWORDS = set([
    'dan', 'yang', 'di', 'ke', 'dari', 'ini', 'itu', 'aku', 'kamu', 'dia', 'saya', 'sangat', 'sayang',
    'kita', 'mereka', 'untuk', 'karena', 'dengan', 'atau', 'tapi', 'tetapi', 'makin', 'semakin',
    'juga', 'sudah', 'belum', 'bisa', 'akan', 'ada', 'tidak', 'tak', 'gak', 'sama', 'apa', 'tau', 'memang', 'oke',
    'bukan', 'saja', 'lagi', 'kok', 'sih', 'dong', 'kan', 'nya', 'ya', 'biar', 'iya',
    'kalo', 'kalau', 'buat', 'bikin', 'jadi', 'cuma', 'cuman', 'dgn', 'yg', 'masih', 'kayak',
    'utk', 'karna', 'kpd', 'motor', 'honda', 'vario', 'review', 'admin', 'kenapa', 'lebih'
])  

def cleaning(text, kamus_tidak_baku):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    normalized_words = [kamus_tidak_baku.get(word, word) for word in words]
    return " ".join(normalized_words)

def split_into_segments(text):
    if not isinstance(text, str): return []
    split_words = ['tapi', 'tp', 'tpi', 'cuma', 'cuman', 'cm', 'sayang', 'syg', 'soalnya', 'soal', 'karena', 'krn', 'karna', 'namun', 'sedangkan', 'padahal', 'walaupun', 'meskipun', 'tros', 'trus', 'terus', 'apalagi', 'kecuali', 'bedanya', 'biar', 'kalo', 'kl', 'kalau']
    text = text.lower()
    text = re.sub(r'[\.\?\!\n,]+', ' ||| ', text)
    pattern = r'\b(' + '|'.join(split_words) + r')\b'
    text = re.sub(pattern, ' ||| ', text)
    return [seg.strip() for seg in text.split(' ||| ') if seg.strip() and len(seg.strip()) > 3]

def chunk_if_long(text, max_tokens=32):
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_tokens: return [text]
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i+max_tokens]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
    return chunks

def classify_segment(text, topic_dict):
    scores = {topic: 0 for topic in topic_dict}
    for topic, keywords in topic_dict.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                scores[topic] += 1
    max_score = max(scores.values())
    return max(scores, key=scores.get) if max_score > 0 else "Lainnya"

def make_prediction(text_input):
    try:
        response = requests.post(api_url, json={"text": str(text_input)})
        
        if response.status_code == 200:
            return response.json() 
        else:
            return {"prediction": "Error", "confidence": 0.0}
            
    except Exception as e:
        return {"prediction": "Connection Error", "confidence": 0.0}
    
def get_top_words(text_series, top_n=10):   
    all_text = " ".join(text_series.astype(str).tolist()).lower()
    all_text = re.sub(r"[^a-z\s]", "", all_text)
    words = all_text.split()
    final_words = []
    for w in words:
        if w in STOPWORDS:
            continue
        if w.endswith('nya'):
            stemmed_word = w[:-3]
            if len(stemmed_word) > 2:
                final_words.append(stemmed_word)
        elif len(w) > 2:
            final_words.append(w)
            
    if not final_words:
        return pd.DataFrame(columns=['word', 'count'])
        
    word_counts = Counter(final_words)
    return pd.DataFrame(word_counts.most_common(top_n), columns=['word', 'count'])

def main():
    st.title('Astra Honda Sentiment Analysis App')
    st.markdown("Unggah file CSV/Excel berisi teks ulasan/komentar untuk dianalisis sentimennya.")
    if 'final_df' not in st.session_state:
        st.session_state.final_df = None
    uploaded_file = st.file_uploader("Upload CSV/Excel File", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv') : 
                df = pd.read_csv(uploaded_file)
            else :
                df = pd.read_excel(uploaded_file)
            st.subheader("Preview Data Asli")
            st.dataframe(df.head(10))

            text_column = st.selectbox(
                "Pilih kolom yang berisi teks/review:",
                df.columns
            )
            if st.button('Mulai Analisis Sentimen'):
                final_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_rows = len(df)
                
                for idx, row in df.iterrows():
                    original_comment = str(row[text_column] )
                    
                    # 1. Segmentasi
                    segments = split_into_segments(original_comment)
                    if not segments: segments = [original_comment]

                    for seg in segments:
                        # 2. Cleaning & Chunking
                        seg_clean = cleaning(seg, kamus_tidak_baku_dict)
                        seg_chunks = chunk_if_long(seg_clean)

                        for chunk in seg_chunks:
                            # 3. Klasifikasi Topik
                            topic = classify_segment(chunk, custom_topic)
                            sentiment_res = make_prediction(chunk)
                            final_results.append({
                                'Original Comments': original_comment,
                                'Segmented Comments' : chunk,
                                'Topic': topic,
                                'Sentiment': sentiment_res['prediction'],
                                'Confidence': sentiment_res['confidence']
                            })
                    
                    # Update Progress setiap baris
                    progress_bar.progress((idx + 1) / total_rows)
                    status_text.text(f"Memproses baris {idx+1} dari {total_rows}...")

                # Buat DataFrame Hasil Akhir
                df_final = pd.DataFrame(final_results)
                st.session_state.final_df = df_final
                st.success("Analisis Selesai!")
          
            if st.session_state.final_df is not None:
                df_final = st.session_state.final_df
                st.divider()
                st.subheader("Hasil Analisis Lengkap")

                def color_row(val):
                    if val == 'Positive': return 'background-color: #d4edda; color: black'
                    elif val == 'Negative': return 'background-color: #f8d7da; color: black'
                    else: return 'background-color: #fff3cd; color: black'

                st.dataframe(df_final.style.applymap(color_row, subset=['Sentiment']))
                
                #Save As CSV
                csv = df_final.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Laporan Lengkap (CSV)",
                    data=csv,
                    file_name="sentiment_report.csv",
                    mime="text/csv"
                )
                # Save As EXCEL
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_final.to_excel(writer, index=False, sheet_name='Sheet1')

                excel_data = buffer.getvalue()
                st.download_button(
                    label="Download Laporan Lengkap (Excel)",
                    data=excel_data,
                    file_name="sentiment_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )    
                st.divider()
                # VISUALISASI PLOT
                st.header("📊 Dashboard Visualisasi")
                color_map = {
                    "Positive": "#28a745", 
                    "Negative": "#dc3545",
                    "Neutral":  "#6c757d"
                    # "Error":    "#000000"
                }
                st.subheader("Total Sentimen Keseluruhan")
                sentiment_counts = df_final['Sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                target_sentiments = ['Positive', 'Negative', 'Neutral']
                sentiment_counts = sentiment_counts[sentiment_counts['Sentiment'].isin(target_sentiments)]

                total_data = sentiment_counts['Count'].sum()
                sentiment_counts['Percentage'] = (sentiment_counts['Count'] / total_data * 100).round(1)
                sentiment_counts['label_text'] = sentiment_counts.apply(
                    lambda x: f"{x['Count']} ({x['Percentage']}%)", axis=1
                )
                fig1 = px.bar(
                    sentiment_counts, 
                    x='Sentiment', 
                    y='Count',
                    color='Sentiment',
                    color_discrete_map=color_map,
                    text='label_text',
                    title="Distribusi Total Sentimen"
                )
                fig1.update_traces(textposition='auto')
                fig1.update_layout(
                    xaxis_title="",
                    yaxis_title="Jumlah Komentar",
                    showlegend=False 
                )
                
                st.plotly_chart(fig1, use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.divider()
                st.subheader("Peringkat Topik Berdasarkan Sentimen")
            
                col_top_pos, col_top_neg = st.columns(2, gap="large")
                
                # --- CHART KIRI: TOPIK POSITIF TERBANYAK ---
                with col_top_pos:
                    st.markdown("### ✅ Topik Paling Banyak Disukai (Positif)")
                    pos_data = df_final[
                            (df_final['Sentiment'] == 'Positive') & 
                            (df_final['Topic'] != 'Lainnya')]
                    
                    if not pos_data.empty:
                        pos_topic_counts = pos_data['Topic'].value_counts().reset_index()
                        pos_topic_counts.columns = ['Topic', 'Count']

                        pos_topic_counts = pos_topic_counts.sort_values(by='Count', ascending=True).tail(10) 
                        fig_p = px.bar(
                            pos_topic_counts, 
                            x='Count', 
                            y='Topic', 
                            orientation='h',
                            text='Count',
                            title="Top Topik Positif",
                            color_discrete_sequence=['#28a745']
                        )
                        fig_p.update_traces(textposition='auto')
                        fig_p.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
                        st.plotly_chart(fig_p, use_container_width=True)
                    else:
                        st.info("Tidak ada data sentimen positif.")

                with col_top_neg:
                    st.markdown("### ❌ Topik Paling Banyak Dikeluhkan (Negatif)")
                    neg_data = df_final[
                        (df_final['Sentiment'] == 'Negative') & (df_final['Topic'] != 'Lainnya')]
                    
                    if not neg_data.empty:
                        neg_topic_counts = neg_data['Topic'].value_counts().reset_index()
                        neg_topic_counts.columns = ['Topic', 'Count']
                        neg_topic_counts = neg_topic_counts.sort_values(by='Count', ascending=True).tail(10)
                        
                        fig_n = px.bar(
                            neg_topic_counts, 
                            x='Count', 
                            y='Topic', 
                            orientation='h',
                            text='Count',
                            title="Top Topik Negatif",
                            color_discrete_sequence=['#dc3545'] # Merah
                        )
                        fig_n.update_traces(textposition='auto')
                        fig_n.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
                        st.plotly_chart(fig_n, use_container_width=True)
                    else:
                        st.info("Tidak ada data sentimen negatif.")

                st.subheader("Sentimen per Topik")
                topic_list = df_final['Topic'].unique().tolist()
                if "Lainnya" in topic_list: 
                    topic_list.remove("Lainnya")
                    topic_list.append("Lainnya")
                            
                selected_topic = st.selectbox("Pilih Topik untuk Dilihat:", topic_list)
                filtered_df = df_final[df_final['Topic'] == selected_topic]
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1: 
                    if not filtered_df.empty:
                        topic_counts = filtered_df['Sentiment'].value_counts().reset_index()
                        topic_counts.columns = ['Sentiment', 'Count']
                        st.subheader(f"Distribusi Sentimen: {selected_topic}")
                        fig2 = px.bar(
                            topic_counts, 
                            x='Sentiment', 
                            y='Count',
                            color='Sentiment',
                            color_discrete_map=color_map,
                            text='Count'
                        )
                        fig2.update_traces(textposition='auto')
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.warning("Tidak ada data untuk topik ini.")

                with col_chart2:
                    topic_data = df_final[df_final['Topic'] == selected_topic]
                    collected_words = set()
                    col_pos, col_neg = st.columns(2, gap="large")
                    st.write("") 
                    st.write("")
                    # --- VISUALISASI KATA POSITIF ---
                    with col_pos:
                        st.markdown("### Top 10 Kata (Positif)")
                        pos_text = topic_data[topic_data['Sentiment'] == 'Positive']['Segmented Comments']
                        
                        if not pos_text.empty:
                            df_pos_words = get_top_words(pos_text)
                            
                            if not df_pos_words.empty:
                                collected_words.update(df_pos_words['word'].tolist())
                                fig_pos = px.bar(
                                    df_pos_words.sort_values(by='count', ascending=True), 
                                    x='count',
                                    y='word',
                                    orientation='h', 
                                    text='count',
                                    color_discrete_sequence=['#28a745'], 
                                    title=f"Kata Positif Terbanyak di {selected_topic}",
                                    height=450,
                                    width=400
                                )
                                # fig_pos.update_layout(plot_bgcolor='white', font=dict(color='black'))
                                fig_pos.update_traces(textposition='auto')
                                st.plotly_chart(fig_pos, use_container_width=True)
                            else:
                                st.info("Tidak cukup kata bermakna untuk dianalisis.")
                        else:
                            st.warning("Tidak ada sentimen positif di topik ini.")

                    with col_neg:
                        st.markdown("### Top 10 Kata (Negatif)")
                        neg_text = topic_data[topic_data['Sentiment'] == 'Negative']['Segmented Comments']
                        
                        if not neg_text.empty:
                            df_neg_words = get_top_words(neg_text)
                            
                            if not df_neg_words.empty:
                                collected_words.update(df_neg_words['word'].tolist())
                                fig_neg = px.bar(
                                    df_neg_words.sort_values(by='count', ascending=True),
                                    x='count',
                                    y='word',
                                    orientation='h',
                                    text='count',
                                    color_discrete_sequence=['#dc3545'], 
                                    title=f"Kata Negatif Terbanyak di {selected_topic}",
                                    height=450,
                                    width=400
                                )
                                # fig_neg.update_layout(plot_bgcolor='white', font=dict(color='black'))
                                fig_neg.update_traces(textposition='auto')
                                st.plotly_chart(fig_neg, use_container_width=True)
                            else:
                                st.info("Tidak cukup kata bermakna untuk dianalisis.")
                        else:
                            st.warning("Tidak ada sentimen negatif di topik ini.")
                ### DRILL DOWN
                st.divider()
                st.markdown(f"### Lihat Konteks Komentar: {selected_topic}")
                
                if collected_words:
                    selected_word_context = st.selectbox(
                        "Pilih kata dari grafik di atas untuk melihat komentar aslinya:",
                        sorted(list(collected_words))
                    )
                    if selected_word_context:
                        context_df = topic_data[topic_data['Segmented Comments'].str.contains(selected_word_context, case=False, na=False)]
                        st.write("Filter Sentimen:")
                        selected_sentiment_filter = st.radio(
                            "Pilih Sentimen:",
                            ["All", "Positive", "Negative", "Neutral"],
                            horizontal=True,
                            label_visibility="collapsed"
                        )

                        if selected_sentiment_filter != "All":
                            context_df = context_df[context_df['Sentiment'] == selected_sentiment_filter]

                        st.markdown("---")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total Komentar", len(context_df))
                        if not context_df.empty:
                            dom_sent = context_df['Sentiment'].mode()[0]
                            m2.metric("Sentimen Dominan", dom_sent)
                            m3.metric("Kata Kunci", selected_word_context)
                        
                        st.markdown("---")
                        st.write(f"Menampilkan komentar yang mengandung kata **'{selected_word_context}'**:")
                        if not context_df.empty:
                            for idx, row in context_df.head(50).iterrows():
                                sentiment = row['Sentiment']
                                text = row['Original Comments']
                                segment = row['Segmented Comments']

                                if sentiment == 'Positive':
                                    border_color = "green"
                                    icon = "✅"
                                elif sentiment == 'Negative':
                                    border_color = "red"
                                    icon = "❌"
                                else:
                                    border_color = "grey"
                                    icon = "😐"

                                highlighted_segment = re.sub(
                                    f"(?i)({re.escape(selected_word_context)})", 
                                    r'<span style="background-color: #ffeeba; padding: 2px 4px; border-radius: 4px; font-weight: bold;">\1</span>',
                                    segment
                                )
                                
                                # Tampilkan Card
                                with st.container():
                                    st.markdown(f"""
                                    <div style="
                                        padding: 15px; 
                                        border-radius: 10px; 
                                        border-left: 5px solid {border_color}; 
                                        background-color: #f9f9f9;
                                        margin-bottom: 10px;
                                        color: black;
                                    ">
                                        <small><strong>{icon} {sentiment}</strong></small><br>
                                        <span style="font-size:16px;">"{highlighted_segment}"</span><br>
                                        <hr style="margin:5px 0;">
                                        <small style="color:grey;"><i>Original: {text}</i></small>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.warning("Tidak ada komentar yang cocok dengan filter ini.")
                            
                    else:
                        st.info("Silakan pilih kata kunci.")
                else:
                    st.info("Belum ada kata kunci yang ditemukan.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")

if __name__ == '__main__':
    main()