import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# ==========================================
# 1. CONFIGURACIÓN Y ESTILO (DARK MODE CREATIVO)
# ==========================================
st.set_page_config(
    page_title="Prompt-Vault | Búsqueda Semántica",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Fira Code', monospace;
        background-color: #0d1117 !important;
        color: #c9d1d9 !important;
    }
    
    h1, h2, h3 { color: #58a6ff !important; }
    
    .stTextArea textarea, .stTextInput input {
        background-color: #161b22 !important;
        color: #58a6ff !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
    }
    
    .stButton > button {
        background-color: #238636 !important;
        color: white !important;
        border: none !important;
        transition: 0.2s !important;
    }
    .stButton > button:hover {
        background-color: #2ea043 !important;
        box-shadow: 0 0 10px rgba(46, 160, 67, 0.4);
    }
    
    .result-card {
        background: #161b22;
        border-left: 4px solid #a371f7;
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MOTOR DE PROCESAMIENTO (NLP)
# ==========================================
# Stemmer en español para reducir palabras a su raíz (ej. "luces" -> "luz")
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# ==========================================
# 3. BASE DE DATOS DE EJEMPLO (ASSETS & PROMPTS)
# ==========================================
default_docs = """Entorno cyberpunk con luces de neón moradas, lluvia ácida, renderizado en Octane Render, 8k, hiperrealista.
Textura generativa abstracta, ondas de sonido fluidas, blanco y negro, alto contraste, ideal para TouchDesigner y Luma Matte.
Personaje 3D estilo cartoon, iluminación de estudio suave, colores pastel, texturas de arcilla, render en Blender Cycles.
Visuales para VJ, túnel fractal infinito, geometría sagrada, colores reactivos al BPM, Resolume Arena, bucle perfecto.
Fotografía de producto, empaque minimalista, iluminación dramática lateral, fondo oscuro, estilo editorial de alta costura.
Interfaz de usuario holográfica, HUD de ciencia ficción, cristales líquidos, Framer Motion, diseño interactivo."""

# ==========================================
# 4. INTERFAZ DE USUARIO
# ==========================================
st.title("🧠 Prompt-Vault: Búsqueda Semántica")
st.markdown("Encuentra el **Asset**, **Prompt** o **Textura** exacta en tu biblioteca utilizando el algoritmo TF-IDF. Busca por conceptos y sensaciones, no por palabras exactas.")

col1, col2 = st.columns([2, 1])

# Manejo del estado para los botones de sugerencias
if 'question' not in st.session_state:
    st.session_state.question = "Necesito visuales para mezclar música en vivo"

with col2:
    st.markdown("### 💡 Búsquedas Sugeridas:")
    st.caption("Haz clic en una idea para buscarla:")
    
    if st.button("Visuales para mezclar música en vivo", use_container_width=True):
        st.session_state.question = "Visuales para mezclar música en vivo"
        st.rerun()
        
    if st.button("Un modelo 3D tierno y amigable", use_container_width=True):
        st.session_state.question = "Un modelo 3D tierno y amigable"
        st.rerun()
        
    if st.button("Máscara de blanco y negro para video", use_container_width=True):
        st.session_state.question = "Máscara de blanco y negro para video"
        st.rerun()
        
    if st.button("Un render futurista y oscuro", use_container_width=True):
        st.session_state.question = "Un render futurista y oscuro"
        st.rerun()

with col1:
    text_input = st.text_area("📂 Tu Biblioteca de Prompts (Un asset por línea):", default_docs, height=180)
    question = st.text_input("🔍 ¿Qué estás buscando hoy?", st.session_state.question)

# ==========================================
# 5. EJECUCIÓN DEL ALGORITMO TF-IDF
# ==========================================
if st.button("Analizar y Buscar 🚀", type="primary"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.error("⚠️ Tu biblioteca está vacía.")
    elif not question.strip():
        st.error("⚠️ Escribe qué estás buscando.")
    else:
        with st.spinner("Vectorizando lenguaje natural..."):
            # Crear vectorizador TF-IDF
            vectorizer = TfidfVectorizer(
                tokenizer=tokenize_and_stem,
                min_df=1 
            )
            
            # Ajustar con documentos
            X = vectorizer.fit_transform(documents)
            
            # Calcular similitud del coseno con la pregunta
            question_vec = vectorizer.transform([question])
            similarities = cosine_similarity(question_vec, X).flatten()
            
            # Encontrar la mejor coincidencia
            best_idx = similarities.argmax()
            best_doc = documents[best_idx]
            best_score = similarities[best_idx]
            
            # 6. MOSTRAR RESULTADOS
            st.markdown("---")
            st.markdown("### 🎯 Match Encontrado")
            
            if best_score > 0.05: 
                st.markdown(f"""
                <div class="result-card">
                    <p style="color: #a371f7; margin-bottom: 5px; font-weight: bold;">Asset / Prompt Recomendado:</p>
                    <h4 style="margin-top: 0; color: white;">{best_doc}</h4>
                    <p style="font-size: 0.8em; color: #8b949e; margin-bottom: 0;">Nivel de Confianza: {best_score*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Acciones rápidas (Copiar)
                st.code(best_doc, language="text")
            else:
                st.warning("No se encontró ninguna coincidencia fuerte en tu biblioteca. Intenta describir el concepto de otra forma.")

            # Modo Ingeniero: Ver las matemáticas detrás (Oculto en un expander para no saturar visualmente)
            with st.expander("⚙️ Ver Matriz Matemática (Pesos TF-IDF)"):
                st.write("Así es como la IA ponderó la importancia de cada palabra en tu biblioteca (las palabras raras valen más que las comunes):")
                df_tfidf = pd.DataFrame(
                    X.toarray(),
                    columns=vectorizer.get_feature_names_out(),
                    index=[f"Asset {i+1}" for i in range(len(documents))]
                )
                st.dataframe(df_tfidf.round(3), use_container_width=True)
