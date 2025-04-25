# import streamlit as st
# from sentence_transformers import SentenceTransformer
# import torch
# from torch.nn.functional import cosine_similarity, pairwise_distance
# import fitz  # PyMuPDF

# # Load model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# st.title("📄 Resume Ranker App")
# st.write("Upload resumes and enter a job description to get ranked matches!")

# similarity_option = st.selectbox("Select Similarity Metric", ["Cosine Similarity", "Euclidean Distance", "Dot Product"])

# uploaded_files = st.file_uploader("Upload Resumes (.txt, .pdf)", type=["txt", "pdf"], accept_multiple_files=True)
# job_description = st.text_area("Paste Job Description")

# def extract_text_from_pdf(pdf_file):
#     doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text

# if uploaded_files and job_description:
#     resume_texts, file_bytes = [], []
#     for file in uploaded_files:
#         file_bytes.append(file.getvalue())
#         if file.type == "application/pdf":
#             resume_texts.append(extract_text_from_pdf(file))
#         else:
#             resume_texts.append(file.read().decode("utf-8"))

#     job_embedding = model.encode(job_description, convert_to_tensor=True)
#     resume_embeddings = model.encode(resume_texts, convert_to_tensor=True)

#     # Similarity scoring
#     if similarity_option == "Cosine Similarity":
#         scores = cosine_similarity(job_embedding, resume_embeddings)
#     elif similarity_option == "Euclidean Distance":
#         scores = -pairwise_distance(job_embedding.unsqueeze(0), resume_embeddings)
#     elif similarity_option == "Dot Product":
#         scores = torch.matmul(resume_embeddings, job_embedding)

#     ranked = torch.argsort(scores, descending=True)

#     st.subheader("🏆 Ranked Resumes:")
#     for idx in ranked:
#         idx = idx.item()
#         file = uploaded_files[idx]
#         score = scores[idx].item()
#         st.markdown(f"### 📌 {file.name} — **Score: {score:.4f}**")

#         description = resume_texts[idx].strip().split('\n')
#         summary = "\n".join(description[:3]) if len(description) >= 3 else resume_texts[idx][:300]
#         st.markdown(f"**Candidate Summary:**\n{summary}")

#         st.download_button(
#             label="📥 Download Resume",
#             data=file_bytes[idx],
#             file_name=file.name,
#             mime=file.type
#         )
#         st.markdown("---")


import streamlit as st
from sentence_transformers import SentenceTransformer
import torch
import fitz  # PyMuPDF
from io import BytesIO
import numpy as np
from sklearn.cluster import DBSCAN
import umap
import pandas as pd
import plotly.express as px

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("📄 Resume Ranker App")
st.write("Upload resumes and enter a job description to get ranked matches!")

uploaded_files = st.file_uploader("Upload Resumes (.txt, .pdf)", type=["txt", "pdf"], accept_multiple_files=True)

job_description = st.text_area("Paste Job Description")

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if uploaded_files and job_description:
    resume_texts = []
    file_bytes = []
    resume_names = []  # List to store resume file names
    for file in uploaded_files:
        file_bytes.append(file.getvalue())  # Keep raw bytes for download
        resume_names.append(file.name)  # Store file names
        if file.type == "application/pdf":
            resume_texts.append(extract_text_from_pdf(file))
        else:
            resume_texts.append(file.read().decode("utf-8"))

    # Create embeddings
    resume_embeddings = model.encode(resume_texts, convert_to_tensor=True)
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=0.3, min_samples=2, metric='cosine')  # You can tweak these params
    cluster_labels = dbscan.fit_predict(resume_embeddings)

    # Perform UMAP for dimensionality reduction
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='cosine')  # Adjusting UMAP params
    umap_embeddings = reducer.fit_transform(resume_embeddings)

    # Visualize clusters with resume names in the hover data
    df_umap = pd.DataFrame(umap_embeddings, columns=['UMAP1', 'UMAP2'])
    df_umap['Cluster'] = cluster_labels
    df_umap['Resume Name'] = resume_names  # Add resume names to the DataFrame

    fig = px.scatter(df_umap, x='UMAP1', y='UMAP2', color='Cluster', title="Resume Clusters", 
                     color_continuous_scale='Viridis', labels={'Cluster': 'Cluster'},
                     hover_data=['Resume Name'])  # Add 'Resume Name' to hover_data

    st.plotly_chart(fig)

    st.subheader("🏆 Ranked Resumes:")

    # Rank resumes based on cosine similarity with job description
    job_embedding = model.encode(job_description, convert_to_tensor=True)
    scores = torch.nn.functional.cosine_similarity(job_embedding, resume_embeddings)
    ranked = torch.argsort(scores, descending=True)

    # Display top 3 ranked resumes with their cluster number
    for idx in ranked[:3]:  # Only top 3 resumes
        idx = idx.item()
        file = uploaded_files[idx]
        score = scores[idx].item()
        cluster_num = df_umap.loc[idx, 'Cluster']  # Get the cluster number of the current resume

        st.markdown(f"### 📌 {file.name} — **Score: {score:.4f}** — **Cluster: {cluster_num}**")

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
