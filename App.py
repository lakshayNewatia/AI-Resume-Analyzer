import streamlit as st
import nltk
import spacy
import secrets
from pdf2image import convert_from_path
import socket   
import platform
import pandas as pd
import base64, random
import time, datetime
import pymysql
import os
import getpass
import geocoder
import io
import plotly.express as px 
from geopy.geocoders import Nominatim
from pdfminer.layout import LAParams, LTTextBox
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from streamlit_tags import st_tags
from PIL import Image
import re

# 1. NLTK DOWNLOADS
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

from pyresparser import ResumeParser
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos

# 2. LOAD SPACY MODEL
nlp = spacy.load("en_core_web_sm")

# 3. AI CLIENT SETUP
from google import genai
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def get_gemini_response(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"AI Error: {e}")
        return "AI Service temporarily unavailable."

# ---------- Helper Functions ----------

def get_csv_download_link(df,filename,text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()      
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()
    converter.close()
    fake_file_handle.close()
    return text

def show_pdf(file_path):
    try:
        images = convert_from_path(file_path, first_page=1, last_page=1)
        if images:
            st.image(images[0], caption="Resume Preview", use_container_width=True)
            with open(file_path, "rb") as f:
                st.download_button(
                    label="📥 Download Full Resume",
                    data=f,
                    file_name=os.path.basename(file_path),
                    mime="application/pdf"
                )
    except Exception as e:
        st.error(f"Error rendering preview: {e}")
        st.info("The preview is unavailable, but you can still download the file above.")

def course_recommender(course_list):
    st.subheader("**Courses & Certificates Recommendations 👨‍🎓**")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course

# ---------- Streamlit Page Config ----------
st.set_page_config(page_title="AI Resume Analyzer", page_icon='./Logo/recommend.png')

# ---------- Main Function ----------
def run():
    img = Image.open('Logo/logo.jpg')
    st.image(img)
    st.sidebar.markdown("# Choose Something...")
    activities = ["User", "About"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)

    if choice == 'User':
        # Collecting Misc Info
        act_name = st.text_input('Name*')
        act_mail = st.text_input('Mail*')
        act_mob  = st.text_input('Mobile Number*')
        sec_token = secrets.token_urlsafe(12)
        host_name = socket.gethostname()
        ip_add = socket.gethostbyname(host_name)
        dev_user = getpass.getuser()
        os_name_ver = platform.system() + " " + platform.release()
        g = geocoder.ip('me')
        latlong = g.latlng
        geolocator = Nominatim(user_agent="http")
        location = geolocator.reverse(latlong, language='en')
        address = location.raw['address']
        city = address.get('city','')
        state = address.get('state','')
        country = address.get('country','')  

        # Upload Resume
        st.markdown("<h5 style='text-align: left;'>Upload Your Resume, And Get Smart Recommendations</h5>", unsafe_allow_html=True)
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
        if pdf_file is not None:
            with st.spinner('Hang On While We Cook Magic For You...'):
                time.sleep(4)

            os.makedirs('./Uploaded_Resumes', exist_ok=True)
            save_image_path = './Uploaded_Resumes/' + pdf_file.name
            pdf_name = pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)

            resume_text = pdf_reader(save_image_path)
            doc = nlp(resume_text)

            # Resume data dict
            resume_data = {"name": "User", "email": None, "mobile_number": None, "skills": [], "no_of_pages": 1}

            # Name extraction
            lines = [line.strip() for line in resume_text.split('\n') if line.strip()]
            blacklist = {'Pandas', 'Numpy', 'Spacy', 'Java', 'React', 'Python', 'Resume', 'CV', 'Page'}
            extracted_name = None
            for line in lines[:3]:
                line_doc = nlp(line)
                for ent in line_doc.ents:
                    if ent.label_ == "PERSON" and ent.text.strip() not in blacklist:
                        extracted_name = ent.text.strip()
                        break
                if extracted_name: break
            if not extracted_name:
                fn = pdf_file.name.split('.')[0]
                fn = re.sub(r'(?i)(resume|cv|final|updated|v\d+|20\d{2}|20\d{1})', '', fn)
                fn = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', fn)
                fn = re.sub(r'(_|-|\.)', ' ', fn)
                extracted_name = ' '.join(fn.split()).title()
            resume_data["name"] = extracted_name or "Candidate"

            # Email & Phone extraction
            email_match = re.search(r'[\w\.-]+@[\w\.-]+', resume_text)
            resume_data["email"] = email_match.group(0) if email_match else None
            phone_match = re.search(r'(\d{3}[-.\s]??\d{3}[-.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-.\s]??\d{4}|\d{10})', resume_text)
            resume_data["mobile_number"] = phone_match.group(0) if phone_match else None

            # Skills extraction
            ds_keyword = ['tensorflow','keras','pytorch','machine learning','deep Learning','flask','streamlit']
            web_keyword = ['react', 'django', 'node jS', 'react js', 'php', 'laravel', 'magento', 'wordpress','javascript', 'angular js', 'C#', 'Asp.net', 'flask']
            android_keyword = ['android','android development','flutter','kotlin','xml','kivy']
            ios_keyword = ['ios','ios development','swift','cocoa','cocoa touch','xcode']
            uiux_keyword = ['ux','adobe xd','figma','zeplin','balsamiq','ui','prototyping','wireframes','storyframes','adobe photoshop','photoshop','editing','adobe illustrator','illustrator','adobe after effects','after effects','adobe premier pro','premier pro','adobe indesign','indesign','wireframe','solid','grasp','user research','user experience']
            n_any = ['english','communication','writing', 'microsoft office', 'leadership','customer management', 'social media']

            all_possible_skills = ds_keyword + web_keyword + android_keyword + ios_keyword + uiux_keyword
            found_skills = [skill for skill in all_possible_skills if skill.lower() in resume_text.lower()]
            resume_data["skills"] = list(set(found_skills))

            # Display Analysis
            st.header("**Resume Analysis 🤘**")
            st.success("Hello "+ resume_data['name'])
            
            st.subheader("🤖 AI Summary")
            with st.spinner('Generating AI Pitch...'):
                pitch_prompt = f"Summarize this resume into a 2-line professional pitch: {resume_text[:2500]}"
                ai_pitch = get_gemini_response(pitch_prompt)
                st.info(ai_pitch)

            st.subheader("**Your Basic info 👀**")
            st.text('Name: '+resume_data['name'])
            st.text('Email: ' + str(resume_data['email']))
            st.text('Contact: ' + str(resume_data['mobile_number']))
            st.text('Resume pages: '+str(resume_data['no_of_pages']))

            # Candidate Level Detection
            cand_level = "Fresher"
            if 'INTERNSHIP' in resume_text.upper():
                cand_level = "Intermediate"
            elif 'EXPERIENCE' in resume_text.upper():
                cand_level = "Experienced"
            st.markdown(f"<h4 style='text-align: left; color: #1ed760;'>Candidate Level: {cand_level}</h4>", unsafe_allow_html=True)

            # Skills Recommendations
            st.subheader("**Skills Recommendation 💡**")
            st_tags(label='### Your Current Skills', text='See our skills recommendation below', value=resume_data['skills'], key='1')

            recommended_skills = []
            reco_field = ''
            rec_course = ''

            for i in resume_data['skills']:
                skill_lower = i.lower()
                if skill_lower in ds_keyword:
                    reco_field = 'Data Science'
                    recommended_skills = ['Data Visualization','Predictive Analysis','Statistical Modeling','Data Mining','Clustering & Classification','Data Analytics','Quantitative Analysis','Web Scraping','ML Algorithms','Keras','Pytorch','Probability','Scikit-learn','Tensorflow',"Flask",'Streamlit']
                    rec_course = course_recommender(ds_course)
                    break
                elif skill_lower in web_keyword:
                    reco_field = 'Web Development'
                    recommended_skills = ['React','Django','Node JS','React JS','php','laravel','Magento','wordpress','Javascript','Angular JS','c#','Flask','SDK']
                    rec_course = course_recommender(web_course)
                    break
                elif skill_lower in android_keyword:
                    reco_field = 'Android Development'
                    recommended_skills = ['Android','Android development','Flutter','Kotlin','XML','Java','Kivy','GIT','SDK','SQLite']
                    rec_course = course_recommender(android_course)
                    break
                elif skill_lower in ios_keyword:
                    reco_field = 'IOS Development'
                    recommended_skills = ['IOS','IOS Development','Swift','Cocoa','Cocoa Touch','Xcode','Objective-C','SQLite','Plist','StoreKit',"UI-Kit",'AV Foundation','Auto-Layout']
                    rec_course = course_recommender(ios_course)
                    break
                elif skill_lower in uiux_keyword:
                    reco_field = 'UI-UX Development'
                    recommended_skills = ['UI','User Experience','Adobe XD','Figma','Zeplin','Balsamiq','Prototyping','Wireframes','Storyframes','Adobe Photoshop','Editing','Illustrator','After Effects','Premier Pro','Indesign','Wireframe','Solid','Grasp','User Research']
                    rec_course = course_recommender(uiux_course)
                    break
                else:
                    reco_field = 'NA'
                    recommended_skills = ['No Recommendations']
                    rec_course = "Not Available"

            st_tags(label='### Recommended skills for you.', text='Recommended skills generated from System', value=recommended_skills, key='2')

            # Resume Tips Video
            st.header("**Bonus Video for Resume Writing Tips💡**")
            st.video(random.choice(resume_videos))
            st.header("**Bonus Video for Interview Tips💡**")
            st.video(random.choice(interview_videos))
            st.balloons()

    # About Section
    elif choice == 'About':
        st.subheader("**About The Tool - AI RESUME ANALYZER**")
        st.markdown('''
            <p align='justify'>
                A tool which parses information from a resume using NLP, finds keywords, clusters them by sectors, and shows recommendations & predictions.
            </p>
            <p align='justify'>
                <b>How to use:</b><br>
                <b>User:</b> Upload resume and let the tool analyze it.<br>
                <b>Feedback:</b> Provide feedback on the tool.<br>
                <b>Admin:</b> Login using username <b>admin</b> and password <b>admin@resume-analyzer</b>.
            </p>
        ''', unsafe_allow_html=True)

# Run the Streamlit app
run()
