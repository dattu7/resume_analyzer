import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import docx2txt
import matplotlib.pyplot as plt
st.set_page_config(page_title="Resume Analyzer",page_icon="🤖")

# Function to calculate scaled score
def calculate_scaled_score(job_description, resumes):
    # Extracting job description features
    job_description_text = ' '.join(job_description)
    documents = [job_description_text] + resumes
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(documents)

    # Calculate cosine similarity between job description and resumes
    similarities = cosine_similarity(word_count_vector)
    scores = similarities[0][1:]

    # Scale scores to range 1-10
    min_score = min(scores)
    max_score = max(scores)
    scaled_scores = [((score - min_score) / (max_score - min_score)) * 9 + 1 for score in scores]

    return scaled_scores

# Function to extract text from docx files
def extract_text_from_docx(file):
    text = docx2txt.process(file)
    return text

# Function to extract experience from resume
def extract_experience(text):
    # Example: Extracting years of experience from text
    # This function needs to be adapted based on the format of the resume
    # For demonstration purposes, let's assume the experience is represented as a number in the text
    import re
    experience = re.findall(r'\d+\s*(?:year|yr|years|yrs)', text.lower())
    if experience:
        return int(experience[0].split()[0])
    else:
        return 0

# Streamlit UI
def main():
    st.title("Resume Analyzer")
    # st.set_page_config(page_title="Dattu's website",page_icon="🌹")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Job Description", "Score", "Visualization"])

    if page == "Home":
        st.markdown("### <span style='color: lightgrey;'>Welcome to Resume Analyzer</span>", unsafe_allow_html=True)
        st.markdown("### <span style='color: skyblue;'>Instructions</span>", unsafe_allow_html=True)
        st.write("This application helps you analyze resumes and find suitable candidates based on job descriptions.")
        st.write("Navigate to 'Job Description' to upload job descriptions and resumes, 'Score' to view results, and 'Visualization' to visualize data.")

    elif page == "Job Description":
        st.header("Upload Job Description and Resumes")
        col1,col2=st.columns(2)
        company = col1.text_input("Company")
        role = col2.text_input("Role")
        skills_required = st.multiselect("Skills Required", ["Python", "SQL", "Machine Learning", "Data Analysis", "TensorFlow", "Pandas", "Scikit-learn", "Java", "C++", "JavaScript", "React", "Node.js", "MongoDB"])
        st.session_state.experience_required = st.number_input("Experience Required (years)")
        num_resumes_to_pick = st.number_input("Number of Resumes to Pick", min_value=1)

        uploaded_files = st.file_uploader("Upload Resumes", accept_multiple_files=True, type=['docx'])

        submit_button = st.button("Submit")
        if submit_button and company and role and skills_required and uploaded_files:
            resumes_content = [extract_text_from_docx(resume) for resume in uploaded_files]
            scaled_scores = calculate_scaled_score((company, role, ' '.join(skills_required)), resumes_content)
            candidate_names = [file.name.split('.')[0] for file in uploaded_files]
            scores_df = pd.DataFrame({
                'Candidate': candidate_names,
                'Score': scaled_scores
            })
            scores_df['Experience'] = [extract_experience(text) for text in resumes_content]
            
            st.session_state.scores_df = scores_df
            st.session_state.num_resumes_to_pick = num_resumes_to_pick
            st.success("Resumes and job description submitted successfully.")
        elif submit_button:
            st.warning("Please fill in all fields and upload resumes.")

    elif page == "Score":
        st.header("Results")
        if "scores_df" not in st.session_state:
            st.warning("Please upload resumes and job description first.")
        else:
            st.write("Scaled Scores:")
            st.write(st.session_state.scores_df)

            st.write("Suitable Resumes:")
            suitable_resumes = st.session_state.scores_df[st.session_state.scores_df['Experience'] >= st.session_state.experience_required].nlargest(st.session_state.num_resumes_to_pick, 'Score')
            st.write(suitable_resumes['Candidate'].tolist())

    elif page == "Visualization":
        st.header("Visualization")
        if "scores_df" not in st.session_state:
            st.warning("Please upload resumes and job description first.")
        else:
            scores_df = st.session_state.scores_df

            st.write("Score Visualization:")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(scores_df['Candidate'], scores_df['Score'], color='skyblue', width=0.5)
            ax.set_xlabel('Candidate')
            ax.set_ylabel('Score (1-10)')
            ax.set_title('Score Visualization')
            st.pyplot(fig)

            st.write("Experience Visualization:")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(scores_df['Candidate'], scores_df['Experience'], color='green', width=0.5)
            ax.axhline(y=st.session_state.experience_required, color='red', linestyle='--', linewidth=2, label='Required Experience')
            ax.set_xlabel('Candidate')
            ax.set_ylabel('Experience (years)')
            ax.set_title('Experience Visualization')
            ax.legend()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
