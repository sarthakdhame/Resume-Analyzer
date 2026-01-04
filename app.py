import os
import pickle
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from docx import Document
from pypdf import PdfReader
import shutil

word_vectorizer = pickle.load(open('tfidf.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Minimal skill bank to surface recognizable keywords in the results table
SKILL_BANK = {
    'python', 'java', 'c++', 'c#', 'javascript', 'typescript', 'react', 'node', 'angular', 'vue',
    'django', 'flask', 'fastapi', 'spring', 'dotnet', '.net', 'asp.net', 'php', 'laravel',
    'sql', 'mysql', 'postgres', 'mongodb', 'redis', 'elasticsearch', 'kafka', 'spark', 'hadoop',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins', 'git',
    'machine learning', 'deep learning', 'pytorch', 'tensorflow', 'nlp', 'data science',
    'power bi', 'tableau', 'excel', 'etl', 'qa', 'selenium', 'cypress', 'playwright'
}

HEURISTIC_RULES = [
    (
        'Web Designing',
        {
            'web developer', 'frontend', 'front-end', 'html', 'css', 'javascript', 'js', 'typescript',
            'react', 'vue', 'angular', 'ui', 'ux', 'bootstrap', 'tailwind', 'next.js', 'nextjs'
        },
    ),
    (
        'Python Developer',
        {'python', 'django', 'flask', 'fastapi'},
    ),
    (
        'Java Developer',
        {'java', 'spring', 'spring boot'},
    ),
]

def cleanResume(text):
    text = re.sub('http\S+\s',' ',text)
    text = re.sub('RT|cc',' ' ,text)
    text = re.sub('#\S+\s', ' ',text)
    text = re.sub('@\s+', ' ',text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7F]+',' ', text)
    text = re.sub('\s+',' ',text)
    return text


def extract_skills(text: str) -> List[str]:
    lowered = text.lower()
    found = []
    for skill in SKILL_BANK:
        if re.search(r'\b' + re.escape(skill) + r'\b', lowered):
            found.append(skill)
    return sorted(set(found))


def top_labels(probabilities: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
    if probabilities is None or probabilities.size == 0:
        return []
    idxs = probabilities.argsort()[::-1][:top_k]
    return [(category_dict.get(int(idx), 'Unknown'), float(probabilities[idx])) for idx in idxs]


def heuristic_category(text: str) -> str:
    lowered = text.lower()
    for label, keywords in HEURISTIC_RULES:
        hits = sum(1 for kw in keywords if kw in lowered)
        if hits >= 2 or ('web developer' in lowered and label == 'Web Designing'):
            return label
    return ''

category_dict = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

def extract_text_from_file(uploaded_file):
    """Extract text from supported resume files (PDF, DOCX)."""
    file_name = uploaded_file.name.lower()

    if file_name.endswith('.pdf'):
        reader = PdfReader(uploaded_file)
        pages_text = [page.extract_text() or '' for page in reader.pages]
        return '\n'.join(pages_text)

    if file_name.endswith('.docx'):
        document = Document(uploaded_file)
        paragraphs = [para.text for para in document.paragraphs]
        return '\n'.join(paragraphs)

    raise ValueError('Unsupported file type')


def categorize_resume(uploaded_files, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    results = []

    for uploaded_file in uploaded_files:
        try:
            text = extract_text_from_file(uploaded_file)
        except Exception as exc:
            results.append({'filename': uploaded_file.name, 'category': 'Failed to read', 'error': str(exc)})
            continue

        cleaned_resume = cleanResume(text)
        if not cleaned_resume.strip():
            results.append({'filename': uploaded_file.name, 'category': 'No text', 'error': 'File had no readable text'})
            continue

        input_features = word_vectorizer.transform([cleaned_resume])
        prediction_id = model.predict(input_features)[0]
        category_name = category_dict.get(prediction_id, 'Unknown')

        proba = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(input_features)[0]
            except Exception:
                proba = None

        label_ranking = top_labels(proba)
        confidence = round(label_ranking[0][1], 3) if label_ranking else None
        alt1 = f"{label_ranking[1][0]} ({label_ranking[1][1]:.3f})" if len(label_ranking) > 1 else None
        alt2 = f"{label_ranking[2][0]} ({label_ranking[2][1]:.3f})" if len(label_ranking) > 2 else None

        heuristic_label = heuristic_category(cleaned_resume)
        if heuristic_label and (category_name == 'Unknown' or (confidence is not None and confidence < 0.35)):
            category_name = heuristic_label

        category_folder = os.path.join(output_directory, category_name)
        os.makedirs(category_folder, exist_ok=True)

        target_path = os.path.join(category_folder, uploaded_file.name)
        with open(target_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        skills = ', '.join(extract_skills(cleaned_resume))

        results.append(
            {
                'filename': uploaded_file.name,
                'category': category_name,
                'confidence': confidence,
                'alt1': alt1,
                'alt2': alt2,
                'skills': skills,
            }
        )

    return pd.DataFrame(results)


def main():
    st.title('Resume Analyzer and Categorizer')

    with st.sidebar:
        st.subheader('Settings')
        output_directory = st.text_input('Output Directory', 'categorized_resumes')
        st.caption('Files will be copied into subfolders per predicted category.')

    uploaded_files = st.file_uploader('Upload Resumes', type=['pdf', 'docx'], accept_multiple_files=True)

    if st.button('Categorize Resumes'):
        if uploaded_files:
            results_df = categorize_resume(uploaded_files, output_directory)
            st.subheader('Per-file results')
            st.dataframe(results_df)

            if not results_df.empty and 'category' in results_df:
                counts = results_df['category'].value_counts().reset_index()
                counts.columns = ['category', 'count']
                st.subheader('Category distribution')
                st.bar_chart(counts.set_index('category'))

            results_csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label='Download Results as CSV',
                data=results_csv,
                file_name='categorized_resumes.csv',
                mime='text/csv',
            )
            st.success('Resumes categorization and processing completed!')
        else:
            st.error('Please upload at least one resume file.')


if __name__ == '__main__':
    main()