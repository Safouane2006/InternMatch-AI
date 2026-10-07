# InternMatch AI

## Live Demo

🚀 Try the application here: [InternMatch AI](https://internmatch-ai-ckip2htflfecf6pqp7tyyq.streamlit.app/)

InternMatch AI is a job recommendation system that matches a student's profile with job opportunities using Natural Language Processing.

## Project goal

The objective is to recommend the most relevant job opportunities based on:

- Student skills
- Job title
- Required skills
- Job description

## Dataset

The project uses job posting data containing:

- Job title
- Company
- Job location
- Job level
- Job type
- Required skills
- Job summary
- Job link

## Project workflow

1. Data loading
2. Data exploration
3. Dataset merging
4. Missing value handling
5. Feature engineering
6. Text cleaning
7. TF-IDF transformation
8. Cosine similarity
9. Job recommendation
10. Streamlit application

## Feature Engineering

The following columns are combined:

- `job_title`
- `job_skills`
- `job_summary`

They are grouped into a new column called `matching_text`.

This text is then cleaned and transformed into numerical features.

## TF-IDF

TF-IDF is used to transform job descriptions and skills into numerical vectors that can be processed by the recommendation system.

## Cosine Similarity

Cosine similarity compares the student's profile vector with all job vectors.

The jobs with the highest similarity scores are recommended to the user.

## Technologies

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Cosine Similarity
- Streamlit

## Application

The Streamlit application allows the user to:

- Describe their profile and skills
- Choose the number of recommendations
- Find the most relevant job opportunities
- View company, location, required skills and matching score

## Run the project

Install the required libraries:

```bash
pip install -r requirements.txt

```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

## Project structure

```text
InternMatch-AI/
├── data/
│   ├── job_postings.csv
│   ├── job_skills.csv
│   └── job_summary.csv
├── notebooks/
│   └── analysis.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## Future improvements

- CV upload and automatic skill extraction
- Filtering by location
- Filtering by job level
- Better NLP embeddings
- Personalized recommendations
- Online deployment