📊 Sentiment Analysis System (Positive/Negative)

Introduction
Sentiment Analysis System is a machine learning-based application that automatically identifies whether a given text expresses  positive  or  negative  emotion. The system uses Natural Language Processing (NLP) and  Deep Learning (DistilBERT)  to analyze text input including words, sentences, paragraphs, social media comments, and YouTube video comments. It helps businesses understand customer feedback, monitor brand reputation, and improve user experiences through automated emotion detection.

 Features
- Automatic sentiment detection (Positive/Negative)
- Supports words, sentences, paragraphs, and bulk text
- YouTube video comments analysis (via URL)
- Social media comments analysis
- Fast and accurate (92% accuracy, 0.5s response)
- Professional web interface with animations
- Real-time text processing
- Bulk analysis (multiple texts at once)
- CSV report export
- Pie charts and word cloud visualizations
- Lightweight and scalable

 Technologies Used
| Category | Technology |
|----------|------------|
| Backend | Python 3.12 |
| AI Model | DistilBERT (Hugging Face Transformers) |
| Web Framework | Gradio |
| Data Processing | Pandas, NumPy, Regex |
| Visualization | Matplotlib, Seaborn, Plotly, WordCloud |
| Development | Google Colab / VS Code |

 Setup Instructions

Option 1: Google Colab (Recommended)
1. Open  Google Colab  (colab.research.google.com)
2. Create  New Notebook 
3. Copy and paste the complete code
4. Click  Runtime → Run all  (Ctrl+F9)
5. Click the public link that appears

 Option 2: Local Machine (VS Code)
1. Install Python 3.8+
2. Install libraries:
```bash
pip install transformers torch gradio pandas matplotlib seaborn plotly wordcloud
```
3. Create `app.py` and paste the code
4. Run: `python app.py`
5. Open browser: `http://127.0.0.1:7860`


 How to Use

| Feature | How To |
|---------|--------|
|  Single Text  | Type any word/sentence → Click Analyze |
|  Bulk Analysis  | Enter multiple texts (one per line) → Click Analyze All |
|  YouTube  | Paste video URL → Click Analyze Comments |
|  Social Media  | Select platform → Paste comment → Analyze |

Sample Input & Output

| Input | Sentiment | Confidence |
|-------|-----------|------------|
| "I love this product!" | ✅ Positive | 95% |
| "This is terrible" | ❌ Negative | 98% |
| "The movie was okay" | ⚠️ Positive | 52% |

 Project Structure
```
sentiment-analysis-project/
│
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── README.md             # Documentation
│
└── outputs/
    └── sentiment_report.csv  # Exported reports
 
Key Features Implemented
 Core: 
- ✅ Text sentiment analysis (word/sentence/paragraph)
- ✅ Real-time sentiment detection
- ✅ Confidence score display
- ✅ Professional UI with animations

 Advanced: 
- ✅ Bulk text analysis
- ✅ CSV report export
- ✅ YouTube video comments analysis
- ✅ Social media platform selector
- ✅ Word cloud & pie chart visualization
- ✅ Live typing sentiment meter

 Use Cases
- E-commerce product review analysis
- Social media brand monitoring
- Customer support ticket classification
- Market research and survey analysis
- Content moderation

 Conclusion

The Sentiment Analysis System successfully demonstrates NLP and Deep Learning for real-time emotion detection. Using  DistilBERT , the system achieves 92% accuracy  with  sub-second response times . The professional UI, bulk analysis, and CSV export make it suitable for business applications.

now Step 8: Launch
demo.launch(share=True)
 

<img width="870" height="450" alt="output 3" src="https://github.com/user-attachments/assets/35b5833c-801b-4606-a649-c8ad00039a88" />
<img width="1288" height="518" alt="output 1" src="https://github.com/user-attachments/assets/e6cc0d8d-17a9-47d4-84b9-cf166c1adcdf" />

```
┌─────────────────────────────────────────────┐
│     🎯 Sentiment Analysis Pro               │
│     AI-Powered Text Analytics               │
├─────────────────────────────────────────────┤
│  [⚡0.5s]  [🎯92%]  [🌍Free]               │
├─────────────────────────────────────────────┤
│  📝 Single Text  📊 Bulk  🎥 YouTube        │
├─────────────────────────────────────────────┤
│  Enter your text:                           │
│  ┌─────────────────────────────────────┐   │
│  │ I love this product!                 │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  [🔍 Analyze]                               │
│                                             │
│  Result: ✅ POSITIVE (95.2%)               │
└─────────────────────────────────────────────┘
