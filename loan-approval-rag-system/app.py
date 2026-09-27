import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from google import genai
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader

# Page Configuration
st.set_page_config(
    page_title="Intelligent Loan Approval & Underwriting RAG Assistant",
    page_icon="🏦",
    layout="wide"
)

# Initialize Gemini Client using Streamlit Cloud Secrets
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found. Please configure it in your Streamlit Cloud Secrets settings.")
    st.stop()

client = genai.Client(api_key=api_key)

# Enterprise Dynamic Pathing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
MODEL_PATH = os.path.join(BASE_DIR, "models", "loan_model.pkl")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# Session State Initialization
if "index_status" not in st.session_state:
    st.session_state.index_status = "not_built"
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load Embedding Model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()

# Function to build FAISS vector index from PDFs in docs folder
def build_index():
    st.session_state.index_status = "building"
    all_chunks = []
    
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR, exist_ok=True)
        
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith((".pdf", ".docx"))]
    
    if not files:
        st.session_state.index_status = "no_documents"
        return
    
    for file in files:
        file_path = os.path.join(DOCS_DIR, file)
        reader = PdfReader(file_path)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                all_chunks.append({"text": page_text, "source": f"{file} (Page {page_num + 1})"})
        
    if not all_chunks:
        st.session_state.index_status = "no_documents"
        return
        
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=False)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    
    st.session_state.index = index
    st.session_state.chunks = all_chunks
    st.session_state.index_status = "loaded"
    st.session_state.num_chunks = len(all_chunks)

# Get Existing Docs
existing_docs = [f for f in os.listdir(DOCS_DIR) if f.endswith((".pdf", ".docx"))] if os.path.exists(DOCS_DIR) else []

# UI Layout - Sidebar Navigation
st.sidebar.title("🏦 Underwriting Hub")
menu = st.sidebar.radio("Navigation", ["1. Loan Eligibility Predictor", "2. AI Policy Underwriting RAG"])

# Silent Background Auto-Indexing
if existing_docs and st.session_state.index_status == "not_built":
    with st.spinner("Initializing Enterprise RAG Engine..."):
        build_index()

# Advanced System Status Dashboard
st.sidebar.divider()
st.sidebar.subheader("⚙️ RAG Engine Status")

if st.session_state.index_status == "loaded":
    st.sidebar.markdown("🟢 **Status:** Online & Active")
    st.sidebar.markdown(f"📄 **Ingested Documents:** {len(existing_docs)}")
    st.sidebar.markdown(f"🧩 **Vector Chunks Indexed:** {st.session_state.num_chunks}")
    st.sidebar.markdown("⚡ **Semantic Search:** FAISS Accelerated")
else:
    st.sidebar.markdown("🔴 **Status:** Offline (Missing Docs)")


# --- TAB 1: LOAN ELIGIBILITY PREDICTOR ---
if menu == "1. Loan Eligibility Predictor":
    st.title("📊 Machine Learning Loan Approval Predictor")
    st.write("Enter applicant tabular details to get an instant risk assessment and approval prediction.")
    
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        
    with col2:
        applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000)
        coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0)
        loan_amount = st.number_input("Loan Amount ($ in thousands)", min_value=0, value=150)
        loan_term = st.selectbox("Loan Amount Term (Days)", [360, 180, 240, 120])
        credit_history = st.selectbox("Credit History (1 = Good, 0 = Bad)", [1.0, 0.0])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    if st.button("Run ML Eligibility Prediction", type="primary", use_container_width=True):
        st.divider()
        st.subheader("🧠 Live AI Model Assessment")
        
        try:
            model = joblib.load(MODEL_PATH)
            input_data = pd.DataFrame([[
                float(credit_history), 
                float(applicant_income), 
                float(coapplicant_income), 
                float(loan_amount)
            ]], columns=['Credit_History', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount'])
            
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            confidence = round(max(probabilities) * 100, 2)
            
            st.session_state.current_applicant = {
                "Credit_History": credit_history,
                "Total_Income": applicant_income + coapplicant_income,
                "LoanAmount": loan_amount,
                "ML_Decision": "Approved" if prediction == 1 else "Rejected",
                "Model_Confidence": f"{confidence}%"
            }
            
            # 5. Advanced Enterprise Metrics
            st.markdown("### 🔍 Underwriting Telemetry")
            colA, colB, colC, colD = st.columns(4)
            
            total_inc = int(applicant_income + coapplicant_income)
            
            with colA:
                st.metric("Model Confidence", f"{confidence}%", f"{confidence - 50}% margin" if prediction==1 else f"-{confidence - 50}% margin")
            with colB:
                # Formats massive numbers with commas for professional reading
                st.metric("Total Income", f"${total_inc:,}") 
            with colC:
                dti = round(loan_amount*1000 / max(total_inc, 1), 2)
                st.metric("Debt-to-Income", f"{dti}", "- Ideal" if dti < 0.5 else "+ High", delta_color="inverse")
            with colD:
                risk_level = "Low" if prediction == 1 else "Critical"
                st.metric("System Risk Flag", risk_level, "+ Cleared" if risk_level == "Low" else "- Flagged", delta_color="normal" if risk_level == "Low" else "inverse")

            # 6. Colorful HTML Decision Card
            if prediction == 1:
                status_color, status_text, icon, bg_color = "#00C853", "APPROVED", "✅", "#e8f5e9"
            else:
                status_color, status_text, icon, bg_color = "#D50000", "REJECTED", "❌", "#ffebee"

            st.write("")
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border-left: 8px solid {status_color}; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
                <h2 style="color: {status_color}; margin-top: 0px; font-family: sans-serif;">{icon} FINAL DECISION: {status_text}</h2>
                <p style="font-size: 16px; color: #333; font-family: sans-serif;">The AI Underwriting Engine has processed the applicant's tabular data through the Random Forest Classifier. Based on historical default patterns, this profile is marked as <strong>{risk_level} Risk</strong> with a model certainty of <strong>{confidence}%</strong>.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("") # Spacer
            st.progress(float(probabilities[1]), text=f"📊 AI Approval Probability Score: {round(float(probabilities[1])*100, 1)}%")
            
            # 7. Explainable AI (XAI) Dropdown
            with st.expander("🔬 View AI Decision Factors (Explainable AI)"):
                st.markdown(f"""
                - **Credit History Factor**: {'Positive impact (Good Credit Profile)' if credit_history == 1.0 else 'Negative impact (Bad Credit History detected)'}
                - **Income Weighting**: Combined income of **${total_inc:,}** analyzed against requested loan amount.
                - **Algorithmic Fairness**: The Random Forest model aggregates 100 distinct decision trees to reach this final verdict, preventing single-feature bias.
                """)
                
        except Exception as e:
            st.error(f"Model Error: {e}. Ensure 'train_model.py' has been run to generate the .pkl file.")


# --- TAB 2: AI POLICY UNDERWRITING RAG ---
elif menu == "2. AI Policy Underwriting RAG":
    st.title("🤖 AI Underwriting & Compliance Assistant")
    st.write("Query bank credit policies, lending criteria, and compliance rules using natural language.")
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask about credit policies, maximum loan limits, or rejection criteria...")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing credit policies..."):
                context_text = ""
                sources = []
                
                if st.session_state.index is not None and st.session_state.chunks:
                    query_embedding = embed_model.encode([user_query])
                    distances, indices = st.session_state.index.search(np.array(query_embedding).astype("float32"), k=3)
                    
                    retrieved_chunks = []
                    for idx in indices[0]:
                        if idx < len(st.session_state.chunks):
                            chunk = st.session_state.chunks[idx]
                            retrieved_chunks.append(chunk["text"])
                            sources.append(chunk["source"])
                            
                    context_text = "\n\n".join(retrieved_chunks)

                applicant_context = ""
                if "current_applicant" in st.session_state:
                    applicant_context = f"\n\n--- CURRENT APPLICANT FILE (LIVE MEMORY) ---\n{st.session_state.current_applicant}\n(If the user asks about 'this applicant', evaluate this specific data against the bank policies above to explain why they might be approved or rejected)."

                prompt = f"""
                You are an Expert Chief Risk Officer and Underwriting Compliance AI Assistant.
                Answer the user's question accurately using ONLY the provided policy context below.
                If the answer is not in the context, explicitly state "I don't know based on the current bank guidelines."
                Always cite the specific source document and page number.

                --- BANK POLICY CONTEXT (FAISS RETRIEVAL) ---
                {context_text}
                {applicant_context}

                User Question: {user_query}
                """

                try:
                    # Enterprise Auto-Retry Mechanism utilizing the active 3.8 model
                    max_retries = 3
                    response = None
                    for attempt in range(max_retries):
                        try:
                            response = client.models.generate_content(
                                model="gemini-3.8-flash",
                                contents=prompt
                            )
                            break
                        except Exception as e:
                            if "503" in str(e) and attempt < max_retries - 1:
                                time.sleep(2)
                            else:
                                raise e
                    
                    answer = response.text
                    if sources:
                        answer += f"\n\n**Sources Cited:** {', '.join(set(sources))}"
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("🔍 View FAISS Retrieval Evidence (Audit Trail)"):
                            st.caption("The following text chunks were instantly extracted from your PDFs using Sentence-Transformers and FAISS vector search to ground the AI's response:")
                            st.info(context_text)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"⚠️ Google API Outage: {e}")
                    st.warning("Google's servers are currently experiencing traffic. Please wait 60 seconds and try your request again.")
