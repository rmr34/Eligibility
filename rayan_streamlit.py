import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import parser
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

# Download necessary NLTK data (if not already done)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import Phrases
from gensim import corpora, models
import streamlit as st
from joblib import load


from langdetect import detect
from googletrans import Translator, LANGUAGES

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score

import warnings
warnings.filterwarnings('ignore')



df5= pd.read_csv('ULYP.csv', encoding='ISO-8859-1')
X = df5.drop('Eligibility ', axis=1)  
y = df5['Eligibility ']  
y = y.replace({'No': 0, 'Yes': 1})
X = X.apply(lambda col: col.astype(str) if col.dtype == 'object' else col)
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numerical_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, X.select_dtypes(include=['float64', 'int64', 'int32']).columns),
    ('cat', categorical_transformer, X.select_dtypes(include=['object','bool']).columns)
])

gbm_model = GradientBoostingClassifier(
learning_rate=0.1,
max_depth=8,
n_estimators=200,
random_state=42
)


























universities = [
    'American University of Beirut (AUB)',
    'Lebanese American University (LAU)',
    'University of Balamand (UOB)',
    'Saint Joseph University of Beirut (USJ)',
    'Université Libanaise (Lebanese University) (LU)',
    'Notre Dame University - Louaize (NDU)',
    'Holy Spirit University of Kaslik (USEK)',
    'Beirut Arab University (BAU)',
    'Lebanese International University (LIU)',
    'American University of Science and Technology (AUST)',
    'Lebanese University of Beirut (UL)',
    'Middle East University (MEU)',
    'Rafik Hariri University (RHU)',
    'Lebanese German University (LGU)',
    'Haigazian University',
    'Not Listed'
]

# Initialize a list to hold data which will eventually be converted to a DataFrame
data = []

# Streamlit page configuration
st.set_page_config(page_title='Scholarship Application Form')

model = load('best_gbmodel.joblib')
lda_model = load('lda_model.joblib')
dictionary = load('dictionary.joblib')
bigram_mod = load('bigram_mod.joblib')
trigram_mod = load('trigram_mod.joblib')

# Creating a form for inputs
with st.form("user_input_form"):
    st.header("Scholarship Application Form")

    # Personal information
    full_name = st.text_input("Full Name")
    sex = st.radio("Sex", options=["Male", "Female"])
    dob = st.date_input("Date of Birth", min_value=datetime(1900, 1, 1), max_value=datetime.now())
    nationality = st.text_input("Nationality")
    year = "2023-2024"
    un_registration = st.radio("Do you have an UNRWA Registration card?", ("Yes", "No"))
    live_in_camp = st.radio("Do you live in a camp?", ("Yes", "No"))
    first_in_family_uni = st.radio("First person in your family to attend university?", ("Yes", "No"))
    mothers_nationality = st.text_input("Mother's Nationality")
    upc_member = st.radio("UPC Member", ("Yes", "No"))
    school_type = st.selectbox("What is the type of your school?", ("Public", "Private", "UNRWA"))
    on_scholarship = st.radio("Are you on a scholarship at your school?", ("Yes", "No"))
    grade_12_stream = st.selectbox("Current grade 12 stream", ("General Sciences - GS", "Life Sciences - LS", "Sociology and Economy - SE", "Literature and Humanities - LH"))

    # Educational background
    grade_10_11_avg = st.number_input("What was your average grade in 10 and 11? (in %)", min_value=0.0, max_value=100.0, format="%.2f")
    eng_score_10 = st.number_input("What was your English score in grade 10 at school? (in %)", min_value=0.0, max_value=100.0, format="%.2f")
    eng_score_11 = st.number_input("What was your English score in grade 11 at school? (in %)", min_value=0.0, max_value=100.0, format="%.2f")
    taken_sat = st.radio("Have you taken the SAT?", ("Yes", "No"))
    sat_attempts_options = {
        0: "Not provided",
        1: "1 time",
        2: "2 times",
        3: "3 times",
        4: "4 times"
    }
    sat_attempts1 = st.selectbox("SAT attempts", options=list(sat_attempts_options.keys()))
    sat_attempts = sat_attempts_options[sat_attempts1]
    highest_sat_score = st.number_input("Highest SAT total score (out of 1600)", min_value=0, max_value=1600)

    # Essays and descriptions
    first_major_choice = st.text_input("What is the first choice of major you would like to study at university?")
    second_major_choice = st.text_input("What is the second choice of major you would like to study at university?")
    preferred_major_reasons = st.text_area("Explain the reasons why you chose your preferred majors in 300 to 500 words.")
    scholarship_qualification = st.text_area("Explain why you think you qualify to be awarded this scholarshipin 300 to 500 words.")
    hobbies_interests = st.text_area("Tell us about your hobbies and interestsin 300 to 500 words.")
    volunteering_experience = st.text_area("Tell us about any volunteering/projects/events/campaigns/workshops/clubs you have been a part of in 300 to 500 words.")
    sibling_scholarship = st.radio("Do you have a sibling who is/was on a ULYP scholarship?", ("Yes", "No"))
    parents_marital_status = st.selectbox("Marital Status of your parents", ("Deceased", "Divorced", "Married", "Single", "Widowed"))
    father_employed = st.radio("Is your father employed?", ("Yes", "No"))
    father_work = st.text_input("Father’s work and position: Place of employment")
    father_annual_salary = st.number_input("Father's Annual Salary in USD")
    mother_employed = st.radio("Is your mother employed?", ("Yes", "No"))
    mother_work = st.text_input("Mother’s work and position: Place of employment")
    mother_annual_salary = st.number_input("Mother's Annual Salary in USD")
    siblings_contribute = st.radio("Do any of your siblings work and contribute part of their income to the house?", ("Yes", "No"))
    financial_help = st.radio("Is there anyone else helping your family financially?", ("Yes", "No"))
    father_assets = st.multiselect("Does your father own Assets/Properties?", ["Nothing", "Apartment", "Savings Account", "Land & Property", "Others"])
    other_assets = ""
    if "Others" in father_assets:
        other_assets = st.text_input("Please specify other assets")
        father_assets.remove("Others")  # Optionally remove "Others" from the list for clarity
        father_assets.append(f"Others: {other_assets}")  # Append the specific assets as part of the "Others" entry
    father_assets_str = ", ".join(father_assets)    
    applicant_assets = st.radio("Do you, as an applicant, have any Assets/Properties?", ("Yes", "No"))
    if applicant_assets == "Yes":
        assets_value = st.number_input("Value of your assets in LBP", min_value=0)
    total_annual_expenses = st.number_input("Total Annual Expenses in USD")
    private_health_insurance = st.radio("Do you have a private health insurance?", ("Yes", "No"))
    life_scholarship = st.radio("Life scholarship Government", ("Done", "Not Done"), index=1)
    tomooh = st.radio("Tomooh", ("Done", "Not Done"), index=1)
    
    # Other detailed questions
    travel_priority = st.radio("Do you want to travel outside of Lebanon for university?", ("Yes", "No"))
    lebanese_universities = st.multiselect("Please indicate the universities in Lebanon you intend to apply to:", universities)

    submit_button = st.form_submit_button("Submit")

if submit_button:
    # Append the data as a row in the list
    data.append([
        full_name, sex, dob, nationality, year, un_registration, live_in_camp,
        first_in_family_uni, mothers_nationality, upc_member, school_type, on_scholarship,
        grade_12_stream, grade_10_11_avg, eng_score_10, eng_score_11, taken_sat, sat_attempts,
        highest_sat_score, preferred_major_reasons, scholarship_qualification, hobbies_interests,
        volunteering_experience, travel_priority, lebanese_universities,first_major_choice, second_major_choice, sibling_scholarship, parents_marital_status,
        father_employed, father_work, father_annual_salary, mother_employed, mother_work, mother_annual_salary,
        siblings_contribute, financial_help, father_assets_str,
        applicant_assets, assets_value if applicant_assets == "Yes" else 0, total_annual_expenses,
        private_health_insurance, life_scholarship, tomooh
    ])

    # Convert list to DataFrame
    df = pd.DataFrame(data, columns=[
        "FULL Name", "Sex", "Date of Birth", "Nationality", "Year ", "Do you have an UNRWA Registration card? ", "Do you live in a camp?",
        " First person in your family to attend university?", "Mother's Nationality", "UPC Member", "What is the type of your school?", "Are you on a scholarship at your school?",
        "Current grade 12 stream", "What was your average grade 10 and 11? Please write in percentile % (for example 78.2%)", "What was your English score in grade 10 at school? Please write in percentile % (for example 79.35%)", "What was your English score in grade 11 at school? Please write in percentile % (for example 79.35%)", "Have you taken the SAT?", "SAT attempts ",
        "Highest SAT total score (out of 1600)", "Explain the reasons why you chose your preferred majors. Link them to how they relate to your personality, knowledge, & aspirations. (Min 300 words - Max 500 words)", "Explain why you think you qualify to be awarded this scholarship and how this opportunity will help you in your current and future plans. How will you contribute to your community after graduation? (Min 300 words - Max 500 words)", "Tell us about your hobbies and interests (Min 300 words - Max 500 words)",
        "Tell us about any volunteering/ projects/events/campaigns/workshops/clubs you have ever been a part of. Tell us what makes you special outside your school work (Min 300 words - Max 500 words)", "Do you want to travel outside of Lebanon for university? Is travel your priority and your parents agree that you travel abroad?", "Please indicate the universities in Lebanon you intend to apply to","What is the first choice of major you would like to study at university?", "What is the second choice of major you would like to study at university?", "Do you have a sibling who is/was on a ULYP scholarship?", "Marital Status of your parents",
        "Is your father employed?", "Please tell us what your father works (Father’s work and position: Place of employment) (name of organization or company). Please follow the example below:", "Father's Annual Salary ", "Is your mother employed?", "Please tell us what your mother work:(Mother’s work and position: Place of employment) (name of organization or company). Please follow the example below:", "Mother's Annual Salary ",
        "Do any of your siblings work and contribute part of their income to the house?", "Is there anyone else helping your family financially?", "Does your father own Assets/Properties. Put check next to the applicable answer or answers:", "Do you, as an applicant, have any Assets/Properties; Land & Property, Saving Accounts, Investments, Others", "If you answered yes to the above question, what is the value/amount in LBP?", "Total Annual Expenses ",
        "Do you have a private health insurance?", " life scholarship Government", "Tomooh"
    ])

    # Remove the 'FULL Name' column
    df = df.drop('FULL Name', axis=1)

    # Create a new column named 'ID' and assign unique IDs for each applicant because names are removed for confidentiality
    df['ID'] = range(1, len(df) + 1)

    # Dictionary with old and new column names
    rename_dict = {
        'What was your average grade 10 and 11? Please write in percentile % (for example 78.2%)': 'Average Grade 10 and 11',
        'What was your English score in grade 9 at school? Please write in percentile % (for example 79.35%)': 'English Grade 9',
        'What was your English score in grade 10 at school? Please write in percentile % (for example 79.35%)': 'English Grade 10',
        'What was your English score in grade 11 at school? Please write in percentile % (for example 79.35%)': 'English Grade 11',
        'Highest SAT total score (out of 1600)': 'SAT Total',
        'Explain the reasons why you chose your preferred majors. Link them to how they relate to your personality, knowledge, & aspirations. (Min 300 words - Max 500 words)': 'Preferred Majors & Reasons',
        'Explain why you think you qualify to be awarded this scholarship and how this opportunity will help you in your current and future plans. How will you contribute to your community after graduation? (Min 300 words - Max 500 words)': 'Why me?',
        'Tell us about your hobbies and interests (Min 300 words - Max 500 words)': 'Hobbies and Interests',
        'Do you want to travel outside of Lebanon for university? Is travel your priority and your parents agree that you travel abroad?': 'Travel or here?',
        'Please indicate the universities in Lebanon you intend to apply to': 'Universities in Lebanon',
        'What is the first choice of major you would like to study at university?': '1st Major',
        'What is the second choice of major you would like to study at university?': '2nd Major',
        'What is the third choice of major you would like to study at university?': '3rd Major',
        'Tell us about any volunteering/ projects/events/campaigns/workshops/clubs you have ever been a part of. Tell us what makes you special outside your school work (Min 300 words - Max 500 words)': 'Volunteering,Projects, and More'
    }

    df.rename(columns=rename_dict, inplace=True)



    df['Date of Birth'] = pd.to_datetime(df['Date of Birth'])
    def adjust_year_in_dob(date_series):
        # Extract the year from the datetime series
        year_series = date_series.dt.year

        # Apply adjustments to the year
        corrected_year_series = year_series.copy()
        corrected_year_series[year_series == 2015] = 2005
        corrected_year_series[year_series == 2020] = 2000
        corrected_year_series[year_series == 2021] = 2001
        corrected_year_series[year_series == 2022] = 2002
        corrected_year_series[year_series == 2023] = 2003

        # Replace the year in the datetime series
        corrected_date_series = pd.to_datetime(corrected_year_series.astype(str) + date_series.dt.strftime('-%m-%d %H:%M:%S'))

        return corrected_date_series

    # Assuming 'Date of Birth' has already been corrected by correct_date and converted to datetime format
    df['Date of Birth'] = adjust_year_in_dob(df['Date of Birth'])
    # create age column
    current_year = pd.Timestamp.now().year
    df['Age'] = current_year - df['Date of Birth'].dt.year
    df['Age'].unique()

    df['Eligible Age'] = df['Age'].apply(lambda x: 'Eligible' if 15 <= x <= 24 else 'Not Eligible')
    # df[(df['Age'] <= 15) | (df['Age'] >= 23)][['Eligible Age', 'Age', 'Date of Birth']]

    df = df.drop(['Age', 'Date of Birth'], axis=1)


    is_lebanese_with_palestinian_mother = (df['Nationality'] == 'Lebanese') & (df["Mother's Nationality"] == 'Palestinian')
    is_palestinian = (df['Nationality'] == 'Palestinian')
    is_syrian = (df['Nationality'] == 'Syrian')

    df['ConsideredNationality'] = np.where(is_lebanese_with_palestinian_mother | is_palestinian | is_syrian, 'Considered', 'Not Considered')

    df = df.drop(['Nationality', 'Mother\'s Nationality'], axis=1)


    # Define the mapping for the replacement
    replacement_map = {
        'No': 0,
        'Yes': 1
    }

    # Apply the mapping to the column
    df['Travel or here?'] = df['Travel or here?'].map(replacement_map)


    def extract_salary(s):
        # Convert input to string and handle missing or NaN values
        s = str(s).lower()  # Convert to lowercase to standardize
        if pd.isnull(s) or s.strip() in ['nan', '']:  # Handle NaN and empty strings
            return "0"
    
        # Normalize large numbers and currency for easier parsing
        normalized_s = (s.replace('millions lebanese lira', ' million l.l')
                        .replace('lebanese lira', 'l.l')
                        .replace('mill', ' million')
                        .replace('thousands', ' thousand')
                        .replace('dollars', ' usd')
                        .replace('lbp', 'l.l')
                        .replace('ll', 'l.l')
                        .replace('l.b.p', 'l.l')
                        .replace('l. l.', 'l.l')
                        .replace('l ', 'l.l')
                        .replace('+', ' and ')
                        .replace('.', ''))
    
        # Define regex pattern to match numbers and their qualifiers
        pattern = r'(\d+(?:\.\d+)?)(?:\s*(million|thousand))?\s*(usd|l\.l)?'
        matches = re.findall(pattern, normalized_s)
    
        total_sum = 0
        currency = ''
        for number_str, magnitude, curr in matches:
            number = float(number_str)
            if 'million' in magnitude:
                number *= 1e6
            elif 'thousand' in magnitude:
                number *= 1e3
    
            # Assuming currency designation should be consistent or first encountered one is used
            if curr and not currency:
                currency = curr.upper() if 'usd' in curr else 'L.L'
    
            total_sum += number
    
        return f"{total_sum} {currency}".strip() if total_sum > 0 else "0"
    
    # Correcting the column name if needed
    df['Cleaned Mother Annual Salary'] = df['Mother\'s Annual Salary '].apply(extract_salary)
    
    def convert_salary(salary):
        if pd.isnull(salary) or salary.strip() in ['nan', '0', '']:  # Handle NaN, zero, and empty strings
            return "0 USD"
    
        try:
            # Attempt to convert the salary to a float, if already in USD, simply return
            salary_float = float(salary.replace(' USD', '').replace(',', ''))
            if ' USD' in salary:
                return f"{salary_float:.2f} USD"
    
            # Convert based on the magnitude of the number
            if salary_float >= 100000:  # Hundreds of thousands or millions
                converted = salary_float / 90000
                return f"{converted:.2f} USD"
            elif salary_float < 10:  # Units
                converted = (salary_float * 1e6) / 90000
                return f"{converted:.2f} USD"
            else:  # Tens or thousands
                return f"{salary_float:.2f} USD"
        except ValueError:
            # In case of any conversion error, return the original string with a note
            return "Conversion Error"
    
    def convert_to_integer(salary):
        # Remove "USD" and any leading/trailing spaces
        salary = salary.replace('USD', '').strip()
    
        try:
            # Convert to float first to handle potential decimal points
            salary_float = float(salary)
    
            # Directly convert the float to int to drop any fractional part
            # This step truncates the decimal without rounding
            return int(salary_float)
    
        except ValueError:
            # Return 0 for any conversion errors or non-numeric values
            return 0
    
    df['Cleaned Mother Annual Salary']= df['Cleaned Mother Annual Salary'].apply(convert_salary).apply(convert_to_integer)

    df['total income (USD)']=df['Father\'s Annual Salary '] + df['Cleaned Mother Annual Salary']

    df = df.drop(['Father\'s Annual Salary ', 'Cleaned Mother Annual Salary','Mother\'s Annual Salary ' ], axis=1)


    conditions = [
        df['total income (USD)'] > df['Total Annual Expenses '],  # Income higher than expenses
        df['total income (USD)'] < df['Total Annual Expenses '],  # Income lower than expenses
        df['total income (USD)'] == df['Total Annual Expenses ']  # Income equal to expenses
    ]

    # Define the corresponding choices for each condition
    choices = ['Income Higher', 'Expenses Higher', 'Income Equal']

    # Use numpy.select to apply conditions and choices to create the new column
    df['IncomeVsExpenses'] = np.select(conditions, choices, default='Unknown')

    df = df.drop(['total income (USD)', 'Total Annual Expenses ' ], axis=1)


    condition = (df["Please tell us what your father works (Father’s work and position: Place of employment) (name of organization or company). Please follow the example below:"].str.contains("UNRWA", case=False, na=False)) & \
                (df["Please tell us what your mother work:(Mother’s work and position: Place of employment) (name of organization or company). Please follow the example below:"].str.contains("UNRWA", case=False, na=False))

    # Apply the condition to create the new column
    df['Both Parents work at UNRWA'] = np.where(condition, 1, 0)

    # df['Both Parents work at UNRWA'].unique()

    df = df.drop(['Please tell us what your father works (Father’s work and position: Place of employment) (name of organization or company). Please follow the example below:',
          'Please tell us what your mother work:(Mother’s work and position: Place of employment) (name of organization or company). Please follow the example below:' ], axis=1)


    specified_majors = ['medicine', 'pharmacy', 'architecture engineering', 'dentistry']

    # Check if either '1st Major' or '2nd Major' is different than the specified majors
    # Using `.str.lower()` to make the comparison case-insensitive
    df['Majors Check'] = ((~df['1st Major'].str.lower().isin(specified_majors)) &
                          (~df['2nd Major'].str.lower().isin(specified_majors)))

    df = df.drop(['1st Major', '2nd Major' ], axis=1)


    # uni_list = [
    #     'American University of Beirut (AUB)',
    #     'Lebanese American University (LAU)',
    #     'University of Balamand (UOB)',
    #     'Saint Joseph University of Beirut (USJ)',
    #     'Université Libanaise (Lebanese University) (LU)',
    #     'Notre Dame University - Louaize (NDU)',
    #     'Holy Spirit University of Kaslik (USEK)',
    #     'Beirut Arab University (BAU)',
    #     'Lebanese International University (LIU)',
    #     'American University of Science and Technology (AUST)',
    #     'Lebanese University of Beirut (UL)',
    #     'Middle East University (MEU)',
    #     'Rafik Hariri University (RHU)',
    #     'Lebanese German University (LGU)',
    #     'Haigazian University'
    # ]

    # special_cases = {
    #     "الجامعة اللبنانية في زحلة": "Lebanese University (LU)",
    #     "الجامعة اللبنانية": "Lebanese University (LU)"
    # }
    # def normalize_university_names(entry):
    #     # Convert non-string entries to strings and handle NaN values
    #     if pd.isnull(entry):
    #         return "Not Listed"  # Or another appropriate value for NaN entries
    #     entry = str(entry)

    #     normalized_names = []

    #     # Handle special cases
    #     for arabic_name, english_name in special_cases.items():
    #         if arabic_name in entry:
    #             entry = entry.replace(arabic_name, english_name)

    #     # Check against the uni_list
    #     for uni in uni_list:
    #         # Using case-insensitive matching
    #         if uni.lower() in entry.lower():
    #             normalized_names.append(uni)

    #     # Join the found university names with ', ' or return a default value if none are found
    #     return ', '.join(normalized_names) if normalized_names else "Not Listed"

    # # Apply the function to your DataFrame column (make sure to replace 'YourColumnName' with the actual column name)
    # df['Universities in Lebanon'] = df['Universities in Lebanon'].apply(normalize_university_names)


    def determine_eligibility(row):
        grade = row['Average Grade 10 and 11']
        school_type = row['What is the type of your school?']

        # Assuming "No" indicates a Public school and "Yes" indicates a Private/UNRWA school
        # Adjust the logic if your mapping of scholarship status to school type is different
        if (grade <= 65 and school_type == 'Public') or (grade <= 75 and (school_type == 'UNRWA'or school_type == 'Private')):
            return 0  # Not eligible
        else:
            return 1  # Eligible

    # Apply the function across the rows
    df['AcademicEligibility'] = df.apply(determine_eligibility, axis=1)


    df['SAT Total'] = df['SAT Total'].fillna(0)

    # Define bins and labels for the categorization
    bins = [-np.inf, 900, 1200, np.inf]  # -inf and inf are used to cover all possible values
    labels = ['Low', 'Avg', 'High']

    # Categorize the 'SAT total' scores
    df['SAT Category'] = pd.cut(df['SAT Total'], bins=bins, labels=labels)

    df = df.drop(['SAT Total' ], axis=1)

    df['Have you taken the SAT?'] = df['Have you taken the SAT?'].str.strip().str.capitalize()
    df['Have you taken the SAT?'].unique()

    df['SAT attempts '] = df['SAT attempts '].str.strip()
    df['SAT attempts '] = df['SAT attempts '].fillna('Not provided')

    df['SAT attempts '].unique()


    df[['Preferred Majors & Reasons',
    'Why me?',
    'Hobbies and Interests', 'Volunteering,Projects, and More']]  = df[['Preferred Majors & Reasons',
    'Why me?',
    'Hobbies and Interests','Volunteering,Projects, and More']].astype(str)
    df[['Preferred Majors & Reasons',
    'Why me?',
    'Hobbies and Interests', 'Volunteering,Projects, and More']]  = df[['Preferred Majors & Reasons',
    'Why me?',
    'Hobbies and Interests', 'Volunteering,Projects, and More']].fillna('Not provided')

    # Initialize the translator
    translator = Translator()

    def detect_and_translate(text):
        try:
            # Detect the language of the text
            lang = detect(text)
            # If the text is in Arabic, translate it to English
            if lang == 'ar':
                translated_text = translator.translate(text, src='ar', dest='en').text
                return translated_text
            # If the text is already in English or any other language, return it as is
            else:
                return text
        except Exception as e:
            # In case of any detection or translation error, return the original text
            print(f"Error: {e}")
            return text

    # Assuming 'Preferred Majors & Reasons' is the column you want to check
    # Apply the detection and translation function to the column
    df['Preferred Majors & Reasons'] = df['Preferred Majors & Reasons'].apply(detect_and_translate)

    # Repeat the process for other columns as needed
    df['Why me?'] = df['Why me?'].apply(detect_and_translate)
    df['Hobbies and Interests'] = df['Hobbies and Interests'].apply(detect_and_translate)
    df['Volunteering,Projects, and More'] = df['Volunteering,Projects, and More'].apply(detect_and_translate)

    def preprocess_text(text):
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        # Convert text to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize text
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

        return " ".join(tokens)

    # Apply preprocessing to each text column
    for column in ['Preferred Majors & Reasons', 'Why me?', 'Hobbies and Interests','Volunteering,Projects, and More']:
        df[column +' perprocessed'] = df[column].apply(preprocess_text)

    def get_sentiment(text):
        return TextBlob(text).sentiment.polarity

    # Apply sentiment analysis to a concatenated text of all three columns
    df['Sentiment'] = df.apply(lambda x: get_sentiment(x['Preferred Majors & Reasons perprocessed'] + ' ' + x['Why me? perprocessed'] + ' ' + x['Hobbies and Interests perprocessed'] + ' ' + x['Volunteering,Projects, and More perprocessed']), axis=1)

    from joblib import load

    # Load the LDA model, Dictionary, Bigram and Trigram Phrasers
    lda_model = load('lda_model.joblib')
    dictionary = load('dictionary.joblib')
    bigram_mod = load('bigram_mod.joblib')
    trigram_mod = load('trigram_mod.joblib')

    # Combine the text from the relevant columns
    combined_texts1 = (df['Preferred Majors & Reasons perprocessed'] + ' ' +
                      df['Why me? perprocessed'] + ' ' +
                      df['Hobbies and Interests perprocessed'] + ' ' +
                      df['Volunteering,Projects, and More perprocessed']).apply(lambda text: text.lower())

    # Tokenize and apply bigrams and trigrams
    texts = [trigram_mod[bigram_mod[word_tokenize(text)]] for text in combined_texts1]

    # Convert texts to the bag-of-words format using the existing dictionary
    corpus_new = [dictionary.doc2bow(text) for text in texts]

    # Use LDA model to get topic distribution for new texts
    topic_distributions = [lda_model[bow] for bow in corpus_new]

    # Optionally, extract the dominant topic for each new text
    dominant_topics = [sorted(topics, key=lambda x: x[1], reverse=True)[0][0] for topics in topic_distributions]

    # If you need to print topics for each new document:
    for doc_topics in topic_distributions:
        print(doc_topics)  # Prints the topic distribution for each document

    keywords = ['volunteering', 'goal', 'confidence', 'believe',
                'community', 'giving back', 'academic excellence', 'volunteer', 'leadership']

    # Adjust the preprocessing function to not break phrases of interest before applying it
    def preprocess_text_for_keywords(text, preserve_phrases=[]):
        # Convert to lowercase and remove punctuation
        text = text.lower().translate(str.maketrans('', '', string.punctuation))

        # Preserve specific phrases by replacing spaces with underscores
        for phrase in preserve_phrases:
            text = text.replace(phrase, phrase.replace(' ', '_'))

        return text

    df['Combined Text'] = df['Preferred Majors & Reasons'] + ' ' + df['Why me?'] + ' ' + df['Hobbies and Interests'] + ' ' + df['Volunteering,Projects, and More perprocessed']
    df['Processed for Keywords'] = df['Combined Text'].apply(lambda x: preprocess_text_for_keywords(x, preserve_phrases=['academic excellence']))

    # Count occurrences of each keyword
    def count_keywords(text):
        # Simple function to count occurrences of each keyword in the text
        counts = {}
        for keyword in keywords:
            # Adjust for phrases turned into single tokens
            adjusted_keyword = keyword.replace(' ', '_')
            counts[keyword] = text.count(adjusted_keyword)
        return counts

    # Apply the keyword counting function
    df['Keyword Counts'] = df['Processed for Keywords'].apply(count_keywords)

    df= df.drop(['Preferred Majors & Reasons perprocessed', 'Why me? perprocessed',
          'Hobbies and Interests perprocessed','Preferred Majors & Reasons', 'Why me?', 'Hobbies and Interests',
                'Combined Text', 'Processed for Keywords'], axis=1)


    def categorize_asset(asset_string):
        if 'Nothing' in asset_string:
            return 'Nothing'
        categories = []
        if any(x in asset_string for x in ['Apartment', 'Land & Property', 'Just the house', 'House + car', 'A small shop', 'Shop Ownership', 'shelter', 'Land is mortgaged']):
            categories.append('Real Estate')
        if any(x in asset_string for x in ['Saving Accounts', 'Investments']):
            categories.append('Financial Assets')
        if 'Car' in asset_string or 'Has a library' in asset_string:
            categories.append('Physical Assets')

        if len(categories) > 1:
            return 'Mixed Assets'
        elif categories:
            return categories[0]
        else:
            return 'Others'

    df['Owned Assests Category'] = df['Does your father own Assets/Properties. Put check next to the applicable answer or answers:'].apply(categorize_asset)

    df['Owned Assests Category'].unique()

    df= df.drop(['Does your father own Assets/Properties. Put check next to the applicable answer or answers:'], axis=1)




    df['Is your mother employed?']= df['Is your mother employed?'].fillna('No')
    df['Is your father employed?']= df['Is your father employed?'].fillna('No')
    df['Average Grade 10 and 11']= df['Average Grade 10 and 11'].fillna(0)
    df['English Grade 10']= df['English Grade 10'].fillna(0)
    df['English Grade 11']= df['English Grade 11'].fillna(0)


    def replace_insurance_status(value):
        # Check if value is NaN (null)
        if pd.isnull(value):
            return 'No'
        # Normalize string (trim and lower case) for consistent comparison
        value = value.strip().lower()
        # Replace based on conditions
        if value == 'no' or value == 'na':
            return 'No'
        else:
            return 'Yes'

    # Apply the function to the specified column
    df['Do you have a private health insurance?'] = df['Do you have a private health insurance?'].apply(replace_insurance_status)
    df[' life scholarship Government']= df[' life scholarship Government'].fillna('No')

    df['Tomooh']= df['Tomooh'].replace({'A':'Done'})
    df['Tomooh']= df['Tomooh'].fillna('No')


    # For the 'Keyword Counts' column, first expand the dictionary into separate columns and rename keys
    def expand_and_rename(row):
        return {f"{key}_count": value for key, value in row.items()}

    keywords_expanded = df['Keyword Counts'].apply(lambda x: expand_and_rename(x) if isinstance(x, dict) else {})
    keywords_df = pd.json_normalize(keywords_expanded)
    df = pd.concat([df.drop('Keyword Counts', axis=1), keywords_df], axis=1)


    df = df.apply(lambda col: col.astype(str) if col.dtype == 'object' else col)

    # Define preprocessing for categorical columns (convert to strings and then apply OneHotEncoding)
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Define preprocessing for numerical columns
    numerical_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])

    # Create the ColumnTransformer for both categorical and numerical columns
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, df.select_dtypes(include=['float64', 'int64', 'int32']).columns),
        ('cat', categorical_transformer, df.select_dtypes(include=['object','bool']).columns)
    ])




    

    # Concatenate X and df for processing
    new_df = pd.concat([X, df], ignore_index=True)
    
    # Fit and transform the new_df using the preprocessor
    df_preprocessed = preprocessor.fit_transform(new_df)
    
    # Splitting the transformed data
    train_features = df_preprocessed[:-1]  # Exclude the last row
    test_features = df_preprocessed[-1:]  # Only the last row



    
    # Train the model
    gbm_model.fit(train_features, y)  # Fit using the corresponding part of 'y'
    
    # Prediction on the test set
    prediction = gbm_model.predict(test_features)

    student_status = "meets the eligibility criteria" if prediction[0] == 1 else "doesn't meet the eligibility criteria"

    # Display the result in Streamlit
    st.write(f"Upon evaluating the application, the system has concluded that the student {full_name} is {student_status} for the scholarship.")

        
