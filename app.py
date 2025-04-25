import streamlit as st
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
import torch
import fitz  # PyMuPDF
from io import BytesIO

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("📄 Resume Ranker App")
st.write("Upload resumes and enter a job description to get ranked matches!")

uploaded_files = st.file_uploader("Upload Resumes (.txt, .pdf)", type=["txt", "pdf"], accept_multiple_files=True)

job_description = st.text_area("Paste Job Description")

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if uploaded_files and job_description:
    resume_texts = []
    file_bytes = []
    for file in uploaded_files:
        file_bytes.append(file.getvalue())  # Keep raw bytes for download
        if file.type == "application/pdf":
            resume_texts.append(extract_text_from_pdf(file))
        else:
            resume_texts.append(file.read().decode("utf-8"))

    job_embedding = model.encode(job_description, convert_to_tensor=True)
    resume_embeddings = model.encode(resume_texts, convert_to_tensor=True)
    scores = cosine_similarity(job_embedding, resume_embeddings)
    ranked = torch.argsort(scores, descending=True)

    st.subheader("🏆 Ranked Resumes:")
    for idx in ranked:
        idx = idx.item()
        file = uploaded_files[idx]
        score = scores[idx].item()
        st.markdown(f"### 📌 {file.name} — **Score: {score:.4f}**")

        # Candidate description: first 3 lines or 300 chars
        description = resume_texts[idx].strip().split('\n')
        summary = "\n".join(description[:3]) if len(description) >= 3 else resume_texts[idx][:300]
        st.markdown(f"**Candidate Summary:**\n{summary}")

        # Download/view file
        st.download_button(
            label="📥 Download Resume",
            data=file_bytes[idx],
            file_name=file.name,
            mime=file.type
        )
        st.markdown("---")
