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
from dotenv import load_dotenv

# ===============================
# Autenticación básica
# ===============================

def autenticar():
    if "autenticado" not in st.session_state:
        st.session_state.autenticado = False

    if not st.session_state.autenticado:
        st.markdown("<h1 style='text-align: center;'>📘 Manu te da la bienvenida</h1>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style='text-align: center;'>
                <img src="https://cdn-icons-png.flaticon.com/512/4712/4712035.png" width="120"/>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align: center; font-size: 18px;'>Por favor, ingresa tus credenciales para acceder a los manuales</p>",
            unsafe_allow_html=True,
        )

        with st.container():
            st.markdown(
                """
                <div style="border: 2px solid #ccc; border-radius: 10px; padding: 30px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); background-color: #f9f9f9;">
                """,
                unsafe_allow_html=True,
            )

            usuario = st.text_input("👤 Usuario")
            clave = st.text_input("🔑 Contraseña", type="password")

            col1, col2, _ = st.columns([1, 1, 2])
            with col1:
                login = st.button("Iniciar sesión")
            with col2:
                cancelar = st.button("Cancelar")

            st.markdown("</div>", unsafe_allow_html=True)

        if cancelar:
            st.stop()

        if login:
            try:
                if (
                    usuario == st.secrets["credenciales"]["usuario"]
                    and clave == st.secrets["credenciales"]["clave"]
                ):
                    st.session_state.autenticado = True
                    st.rerun()
                else:
                    st.error("❌ Usuario o contraseña incorrectos")
                    st.stop()
            except Exception:
                st.error("⚠️ No hay credenciales configuradas en secrets.")
                st.stop()

        st.stop()


# ===============================
# Helpers de feedback (PLACEHOLDER)
# ===============================

def ui_feedback_ticket_placeholder(tema: str, pregunta: str, respuesta: str):
    """
    Flujo de feedback 👍/👎 y creación de ticket (sin enviar nada).
    El botón Enviar solo confirma en pantalla.
    Estados: idle | ask | form | done
    """
    if "feedback_stage" not in st.session_state:
        st.session_state["feedback_stage"] = "idle"

    col_pos, col_neg = st.columns(2)
    with col_pos:
        if st.button("👍 Útil", key="fb_up"):
            st.success("¡Gracias por tu feedback! 🙌")
            st.session_state["feedback_stage"] = "idle"
    with col_neg:
        if st.button("👎 Necesité más ayuda", key="fb_down"):
            st.session_state["feedback_stage"] = "ask"

    if st.session_state["feedback_stage"] == "ask":
        st.info(
            "**Siento no haber ayudado lo suficiente.**\n\n¿Quieres generar un ticket solicitando ayuda en base a tu pregunta?"
        )
        col_si, col_no = st.columns(2)
        with col_si:
            if st.button("Sí, generar ticket", key="fb_yes"):
                st.session_state["feedback_stage"] = "form"
        with col_no:
            if st.button("No, gracias", key="fb_no"):
                st.session_state["feedback_stage"] = "idle"
                st.success("Entendido. Seguimos mejorando 💪")

    if st.session_state["feedback_stage"] == "form":
        st.subheader("Generar ticket de ayuda")
        st.caption("Revisa el detalle. Puedes editar el mensaje antes de enviar.")

        st.text_input("Tema", value=tema, disabled=True)
        st.text_area("Tu pregunta", value=pregunta, height=100, disabled=True)

        tienda = st.text_input("Tienda (número o nombre)", placeholder="Ej: 123 o Mall Centro")

        texto_sugerido = (
            f"Desde la tienda # {tienda or '___'} requerimos ayuda para entender de mejor manera "
            f"o aclarar dudas respecto al tema y pregunta adjunta, en donde la respuesta no fue "
            f"aclaratoria del todo. Muchas gracias."
        )
        cuerpo_editable = st.text_area(
            "Mensaje para el equipo de soporte", value=texto_sugerido, height=160
        )

        incluir_contexto = st.checkbox(
            "Incluir la última respuesta de Manu como contexto", value=True
        )

        if st.button("Enviar", type="primary", key="fb_send"):
            # No enviar nada: solo confirmación visual
            st.session_state["feedback_stage"] = "done"
            st.success("✅ (Simulación) Ticket registrado. Aquí no se envía nada todavía.")

    if st.session_state["feedback_stage"] == "done":
        st.info("Tu solicitud fue registrada. Te contactarán pronto (simulación).")


# ===============================
# App principal
# ===============================

autenticar()

# API key opcional vía .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Inicialización de modelos y embeddings
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
embedding = OpenAIEmbeddings()

# Temas → nombre de carpeta FAISS
DOCUMENTOS = {
    "Traspasos": "traspasos",
    "Reserva de presupuestos": "reserva_de_presupuestos",
    "Cierre de día": "cierre_de_dia",
    "Manifiesto digital": "manifiesto_digital",
}

# Frases especiales
SALUDOS = ["hola", "buenos días", "buenas tardes", "buenas", "qué tal", "como estás"]
DESPEDIDAS = ["adios", "chao", "hasta luego", "nos vemos", "hasta pronto"]
AGRADECIMIENTOS = ["gracias", "muchas gracias", "buena manu"]
RECLAMOS = ["no me sirves", "respuesta mala", "respuestas malas", "esto no sirve", "no ayuda", "pésimo"]

# Métricas simples

def registrar_interaccion(pregunta, respuesta, tema, tipo_respuesta="normal"):
    with open("metricas.csv", mode="a", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow([
            datetime.now().isoformat(),
            tema,
            pregunta,
            tipo_respuesta,
            respuesta,
        ])


@st.cache_resource(show_spinner=False)
def cargar_vectorstore(nombre_tema):
    ruta = os.path.join("vectorstores", nombre_tema)
    if not os.path.exists(ruta):
        st.error(f"No se encontró el vectorstore para: {nombre_tema}")
        st.stop()
    return FAISS.load_local(ruta, OpenAIEmbeddings(), allow_dangerous_deserialization=True)


st.title("🤖 Manu: Tu asistente inteligente de manuales para tiendas")

# Estados iniciales
if "tema" not in st.session_state:
    st.session_state.tema = None
    st.session_state.qa = None
    st.session_state.chat_history = []  # lista de (pregunta, respuesta)
    st.session_state.memory = None

# Selección de tema
if st.session_state.tema is None:
    tema = st.selectbox("¿Sobre qué tema necesitas ayuda?", list(DOCUMENTOS.keys()))
    if st.button("Seleccionar tema"):
        with st.spinner("Cargando tema y preparando a Manu..."):
            archivo_vectorstore = DOCUMENTOS[tema]
            vectorstore = cargar_vectorstore(archivo_vectorstore)
            memoria = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm, retriever=vectorstore.as_retriever(), memory=memoria
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
                respuesta = (
                    "Lo siento, puedes contactarte con mis creadores a través de un ticket y "
                    "contarles tu mala experiencia para poder mejorar. Espero poder ser útil la próxima vez."
                )
                tipo = "especial"
            else:
                respuesta = st.session_state.qa.run(pregunta)

            end = time.time()
            respuesta_con_firma = respuesta + "\n\n— Manu 🤖"

            # Guardamos intercambio y reseteamos feedback para nueva pregunta
            st.session_state.chat_history.append((pregunta, respuesta_con_firma))
            st.session_state["ultima_pregunta"] = pregunta
            st.session_state["ultima_respuesta"] = respuesta_con_firma
            st.session_state["feedback_stage"] = "idle"

            registrar_interaccion(pregunta, respuesta, st.session_state.tema, tipo)
            st.markdown(f"⏱️ Tiempo de respuesta: {round(end - start, 2)} segundos")

    # Render del historial (últimos primero)
    for i, (pregunta_chat, respuesta_chat) in enumerate(reversed(st.session_state.chat_history)):
        with st.chat_message("user"):
            st.markdown(pregunta_chat)
        with st.chat_message("assistant"):
            st.markdown(respuesta_chat)

        # Solo mostramos el bloque de feedback debajo del ÚLTIMO intercambio
        if i == 0:
            ui_feedback_ticket_placeholder(
                tema=st.session_state.tema,
                pregunta=pregunta_chat,
                respuesta=respuesta_chat,
            )

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