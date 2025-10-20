import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import warnings

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data files"""
    # Download punkt_tab (newer version) or punkt (older version)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)

# Call the download function
download_nltk_data()

# Page configuration
st.set_page_config(
    page_title="Annual Report Analyzer",
    page_icon="📊",
    layout="wide"
)

# Simple CSS styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric > div {
        color: #2c3e50; /* Make metric text dark */
    }
    .stMetric > label {
        color: #555555; /* Make metric label a bit lighter */
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

def extract_financial_keywords(text):
    """Extract financial keywords from text"""
    financial_keywords = [
        'revenue', 'profit', 'growth', 'ebitda', 'margin', 'investment', 
        'turnover', 'expense', 'cash', 'debt', 'equity', 'dividend',
        'shareholder', 'return', 'roi', 'performance', 'market', 'sales'
    ]
    text_lower = text.lower()
    found_keywords = {}
    for kw in financial_keywords:
        count = len(re.findall(r'\b' + kw + r'\b', text_lower))
        if count > 0:
            found_keywords[kw] = count
    return found_keywords

def calculate_readability(text, sentences, words):
    """Calculate basic readability metrics"""
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    complex_words = [w for w in words if len(w) > 6]
    complex_word_ratio = len(complex_words) / len(words) if words else 0
    reading_ease = 206.835 - 1.015 * avg_sentence_length - 84.6 * complex_word_ratio
    return {
        'avg_sentence_length': avg_sentence_length,
        'complex_word_ratio': complex_word_ratio * 100,
        'reading_ease': max(0, min(100, reading_ease))
    }

# Main App
st.title("📊 Annual Report Analyzer")
st.markdown("Upload a PDF annual report to analyze its content")

# Sidebar
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload PDF Report", type=["pdf"])
    st.markdown("---")
    n_topics = st.slider("Number of Topics", 3, 10, 5)
    top_n_words = st.slider("Top Words to Display", 10, 30, 15)

# Main content
if not uploaded_file:
    st.info("👈 Please upload a PDF file to begin analysis")
    st.markdown("### Features:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📈 Sentiment Analysis**\nAnalyze overall tone and sentiment")
    with col2:
        st.markdown("**☁️ Word Frequency**\nIdentify key terms and patterns")
    with col3:
        st.markdown("**📌 Topic Modeling**\nDiscover main themes")
else:
    # Extract text and suppress warnings from pdfplumber
    with st.spinner("Extracting text from PDF..."):
        all_text = ""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pdfplumber.open(uploaded_file) as pdf:
                total_pages = len(pdf.pages)
                for page in pdf.pages:
                    txt = page.extract_text()
                    if txt:
                        all_text += txt + " "
    
    # Process text
    clean = clean_text(all_text)
    sentences = sent_tokenize(all_text)
    words = all_text.split()
    
    # Overview Metrics
    st.header(" Document Overview")
    
    # Calculate metrics
    avg_sentiment = TextBlob(all_text).sentiment.polarity
    unique_words = len(set(clean.split()))
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Pages", total_pages)
    col2.metric("Word Count", f"{len(words):,}")
    col3.metric("Sentiment", f"{avg_sentiment:.2f}")
    col4.metric("Sentences", f"{len(sentences):,}")
    
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " Sentiment", " Words", " Topics", " Keywords", " Readability"
    ])
    
    # Tab 1: Sentiment Analysis
    with tab1:
        st.subheader("Sentiment Analysis")
        
        # Calculate sentiment for all sentences
        sentiments = []
        for sent in sentences:
            blob = TextBlob(sent)
            sentiments.append(blob.sentiment.polarity)
        
        avg_polarity = np.mean(sentiments)
        positive_count = len([s for s in sentiments if s > 0])
        
        col1, col2 = st.columns(2)
        
        col1.metric("Average Polarity", f"{avg_polarity:.3f}")
        col1.metric("Positive Sentences", f"{positive_count}/{len(sentiments)}")
        
        with col2:
            # Sentiment distribution
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(sentiments, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
            ax.set_xlabel("Sentiment Polarity")
            ax.set_ylabel("Frequency")
            ax.set_title("Sentiment Distribution")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        # Most positive and negative sentences
        st.markdown("**Notable Sentences:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("**Most Positive:**")
            max_idx = sentiments.index(max(sentiments))
            st.write(sentences[max_idx])
        
        with col2:
            st.error("**Most Negative:**")
            min_idx = sentiments.index(min(sentiments))
            st.write(sentences[min_idx])
    
    # Tab 2: Word Analysis
    with tab2:
        st.subheader("Word Frequency Analysis")
        
        # Word cloud
        col1, col2 = st.columns([2, 1])
        
        with col1:
            wc = WordCloud(width=800, height=400, background_color='white',
                           colormap='viridis').generate(clean)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Top words
            tokens = clean.split()
            freq = pd.Series(tokens).value_counts().head(top_n_words)
            
            st.markdown(f"**Top {top_n_words} Words:**")
            for word, count in freq.items():
                st.write(f"• {word}: {count}")
        
        # Bar chart of top words
        st.markdown("**Word Frequency Chart:**")
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(freq))
        ax.barh(y_pos, freq.values, color='#2ecc71', edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(freq.index)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Words")
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Tab 3: Topic Modeling
    with tab3:
        st.subheader("Topic Modeling (LDA)")
        
        with st.spinner("Analyzing topics..."):
            # Perform LDA
            vectorizer = CountVectorizer(max_features=500, stop_words='english')
            doc_term_matrix = vectorizer.fit_transform([clean])
            
            lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda_model.fit(doc_term_matrix)
            
            feature_names = vectorizer.get_feature_names_out()
        
        # Display topics
        cols = st.columns(2)
        for idx, topic in enumerate(lda_model.components_):
            col_idx = idx % 2
            with cols[col_idx]:
                st.markdown(f"**Topic {idx + 1}:**")
                top_words_idx = topic.argsort()[-8:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                
                # Create bar chart for topic
                fig, ax = plt.subplots(figsize=(6, 4))
                y_pos = np.arange(len(top_words))
                ax.barh(y_pos, topic[top_words_idx], color='#9b59b6', edgecolor='black')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_words)
                ax.invert_yaxis()
                ax.set_xlabel("Weight")
                ax.grid(True, axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    # Tab 4: Financial Keywords
    with tab4:
        st.subheader("Financial Keywords Analysis")
        
        financial_kw = extract_financial_keywords(all_text)
        
        if financial_kw:
            fin_df = pd.DataFrame(list(financial_kw.items()), 
                                 columns=['Keyword', 'Frequency'])
            fin_df = fin_df.sort_values('Frequency', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                y_pos = np.arange(len(fin_df))
                ax.barh(y_pos, fin_df['Frequency'].values, color='#e74c3c', edgecolor='black')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(fin_df['Keyword'].values)
                ax.invert_yaxis()
                ax.set_xlabel("Frequency")
                ax.set_ylabel("Keywords")
                ax.set_title("Financial Keyword Frequency")
                ax.grid(True, axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.dataframe(fin_df, use_container_width=True, hide_index=True)
            
            # Context analysis
            st.markdown("**Keyword Context:**")
            selected_kw = st.selectbox("Select keyword to see context", 
                                       fin_df['Keyword'].tolist())
            
            kw_sentences = [s for s in sentences if selected_kw.lower() in s.lower()]
            st.write(f"Found in {len(kw_sentences)} sentences")
            
            if kw_sentences:
                for i, sent in enumerate(kw_sentences[:3], 1):
                    with st.expander(f"Example {i}"):
                        st.write(sent)
        else:
            st.warning("No financial keywords found in the document.")
    
    # Tab 5: Readability
    with tab5:
        st.subheader("Readability Analysis")
        
        readability = calculate_readability(all_text, sentences, words)
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Avg Sentence Length", 
                    f"{readability['avg_sentence_length']:.1f} words")
        col2.metric("Complex Words", 
                    f"{readability['complex_word_ratio']:.1f}%")
        
        reading_ease = readability['reading_ease']
        col3.metric("Reading Ease", f"{reading_ease:.0f}/100")
        
        # Reading ease interpretation
        if reading_ease > 70:
            level = "Easy to read"
            color = "green"
        elif reading_ease > 50:
            level = "Fairly easy"
            color = "blue"
        elif reading_ease > 30:
            level = "Difficult"
            color = "orange"
        else:
            level = "Very difficult"
            color = "red"
        
        st.markdown(f"**Reading Level:** :{color}[{level}]")
        
        # Sentence length distribution
        sentence_lengths = [len(s.split()) for s in sentences]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(sentence_lengths, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        mean_length = np.mean(sentence_lengths)
        ax.axvline(mean_length, color='red', linestyle='--', linewidth=2,
                   label=f'Average: {mean_length:.1f}')
        ax.set_xlabel("Sentence Length (words)")
        ax.set_ylabel("Frequency")
        ax.set_title("Sentence Length Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Export section
    st.markdown("---")
    st.subheader("Export Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Summary report
        summary = f"""
Annual Report Analysis Summary
{'='*40}

Document Statistics:
- Pages: {total_pages}
- Words: {len(words):,}
- Sentences: {len(sentences):,}
- Unique Words: {len(set(clean.split())):,}

Sentiment:
- Average Polarity: {avg_sentiment:.3f}

Readability:
- Reading Ease: {reading_ease:.1f}
- Avg Sentence Length: {readability['avg_sentence_length']:.1f}

Top Keywords:
{chr(10).join([f"- {kw}: {count}" for kw, count in list(financial_kw.items())[:5]])}
        """
        
        st.download_button(
            "Download Summary",
            summary,
            "analysis_summary.txt",
            "text/plain"
        )
    
    with col2:
        if financial_kw:
            fin_df = pd.DataFrame(list(financial_kw.items()), 
                                 columns=['Keyword', 'Frequency'])
            st.download_button(
                "Download Keywords (CSV)",
                fin_df.to_csv(index=False),
                "keywords.csv",
                "text/csv"
            )
    
    st.success("Analysis complete!")