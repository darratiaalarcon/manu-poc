import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
import os
import csv
import time
from datetime import datetime
import os
from dotenv import load_dotenv
from pathlib import Path

# Configura tu API key aquí o como variable de entorno
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key 


# Inicialización de modelos y embeddings
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
embedding = OpenAIEmbeddings()

# Diccionario de temas y archivos (nombre de carpeta FAISS generado previamente)
DOCUMENTOS = {
    "Traspasos": "traspasos",
    "Reserva de presupuestos": "reserva_de_presupuestos",
    "Cierre de día": "cierre_de_dia",
    "Manifiesto digital": "manifiesto_digital"
}

# Diccionarios de respuestas especiales
SALUDOS = ["hola", "buenos días", "buenas tardes", "buenas", "qué tal", "como estás"]
DESPEDIDAS = ["adios", "chao", "hasta luego", "nos vemos", "hasta pronto"]
AGRADECIMIENTOS = ["gracias", "muchas gracias", "buena manu"]
RECLAMOS = ["no me sirves", "respuesta mala", "respuestas malas", "esto no sirve", "no ayuda", "pésimo"]

# Registro de métricas en CSV
def registrar_interaccion(pregunta, respuesta, tema, tipo_respuesta="normal"):
    with open("metricas.csv", mode="a", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow([
            datetime.now().isoformat(),
            tema,
            pregunta,
            tipo_respuesta,
            respuesta
        ])

# Cargar FAISS ya persistido en disco
@st.cache_resource(show_spinner=False)
def cargar_vectorstore(nombre_tema):
    ruta = os.path.join("vectorstores", nombre_tema)
    if not os.path.exists(ruta):
        st.error(f"No se encontró el vectorstore para: {nombre_tema}")
        st.stop()
    return FAISS.load_local(ruta, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

st.title("🤖 Manu: Tu asistente inteligente de manuales para tiendas")

# Inicialización de estados
if "tema" not in st.session_state:
    st.session_state.tema = None
    st.session_state.qa = None
    st.session_state.chat_history = []
    st.session_state.memory = None

# Menú de selección de tema
if st.session_state.tema is None:
    tema = st.selectbox("¿Sobre qué tema necesitas ayuda?", list(DOCUMENTOS.keys()))
    if st.button("Seleccionar tema"):
        with st.spinner("Cargando tema y preparando a Manu..."):
            archivo_vectorstore = DOCUMENTOS[tema]
            vectorstore = cargar_vectorstore(archivo_vectorstore)
            memoria = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memoria
            )
            st.session_state.tema = tema
            st.session_state.qa = qa_chain
            st.session_state.chat_history = []
            st.session_state.memory = memoria
        st.rerun() 
else:
    st.markdown(f"### Tema seleccionado: {st.session_state.tema}")
    pregunta = st.text_input("Haz tu pregunta sobre este tema:")
    if pregunta:
        pregunta_lower = pregunta.strip().lower()
        tipo = "normal"

        with st.spinner("Manu está pensando..."):
            start = time.time()

            if any(s in pregunta_lower for s in SALUDOS):
                respuesta = "¡Muy bien! Estoy aquí para ayudarte 😊"
                tipo = "especial"
            elif any(s in pregunta_lower for s in DESPEDIDAS):
                respuesta = "Adiós, fue un placer ayudarte. Si necesitas algo más, estaré atento 👋"
                tipo = "especial"
            elif any(s in pregunta_lower for s in AGRADECIMIENTOS):
                respuesta = "¡De nada! Estoy muy feliz de haber ayudado 😄"
                tipo = "especial"
            elif any(s in pregunta_lower for s in RECLAMOS):
                respuesta = "Lo siento, puedes contactarte con mis creadores a través de un ticket y contarles tu mala experiencia para poder mejorar. Espero poder ser útil la próxima vez."
                tipo = "especial"
            else:
                respuesta = st.session_state.qa.run(pregunta)

            end = time.time()
            respuesta_con_firma = respuesta + "\n\n— Manu 🤖"
            st.session_state.chat_history.append((pregunta, respuesta_con_firma))
            registrar_interaccion(pregunta, respuesta, st.session_state.tema, tipo)
            st.markdown(f"⏱️ Tiempo de respuesta: {round(end - start, 2)} segundos")

    for pregunta_chat, respuesta_chat in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(pregunta_chat)
        with st.chat_message("assistant"):
            st.markdown(respuesta_chat)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔁 Volver al menú de temas"):
            st.session_state.tema = None
            st.session_state.qa = None
            st.session_state.chat_history = []
            st.session_state.memory = None
            st.rerun()
    with col2:
        if st.button("🧼 Reiniciar conversación actual"):
            st.session_state.chat_history = []
            if st.session_state.memory:
                st.session_state.memory.clear()
            st.success("✅ Conversación reiniciada. Puedes comenzar de nuevo.")
            st.rerun() 
