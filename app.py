from flask import Flask, render_template, request, redirect
import pandas as pd
import plotly.express as px
from plotly.offline import plot
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import plotly.graph_objects as go
import tensorflow_hub as hub
from top2vec import Top2Vec
import re
import string


categories = [
    "ARTS","ARTS & CULTURE","BLACK VOICES","BUSINESS","COLLEGE","COMEDY","CRIME","CULTURE & ARTS","DIVORCE","EDUCATION","ENTERTAINMENT","ENVIRONMENT","FIFTY","FOOD & DRINK","GOOD NEWS","GREEN","HEALTHY LIVING","HOME & LIVING","IMPACT","LATINO VOICES","MEDIA","MONEY","PARENTING","PARENTS","POLITICS","QUEER VOICES","RELIGION","SCIENCE","SPORTS","STYLE","STYLE & BEAUTY","TASTE","TECH","THE WORLDPOST","TRAVEL","U.S. NEWS","WEDDINGS","WEIRD NEWS","WELLNESS","WOMEN","WORLD NEWS","WORLDPOST",
]


dict_categories = {
    "Health": ["WELLNESS", "HEALTHY LIVING"],
    "Entertainment": ["ENTERTAINMENT", "MEDIA", "COMEDY", "ARTS & CULTURE", "CULTURE & ARTS", "ARTS", "QUEER VOICES"],
    "Home": ["WOMEN", "PARENTS", "PARENTING", "HOME & LIVING", "WEDDINGS"],
    "Style": ["STYLE & BEAUTY", "STYLE"],
    "Environment": ["GREEN", "ENVIRONMENT"],
    "Business": ["MONEY", "BUSINESS", "FIFTY"],
    "Positive News": ["GOOD NEWS"],
    "Politics": ["POLITICS", "BLACK VOICES", "LATINO VOICES"],
    "War": ["IMPACT"],
    "Travel": ["TRAVEL"],
    "FOOD": ["FOOD & DRINK", "TASTE"],
    "Academics": ["EDUCATION", "COLLEGE"],
    "Technology": ["TECH", "SCIENCE"],
    "Negative News": ["DIVORCE"],
    "World News": ["WORLDPOST", "U.S. NEWS", "WORLD NEWS", "THE WORLDPOST"],
    "Sports": ["SPORTS"],
    "Religion": ["RELIGION"],
    "Crime": ["CRIME"],
}

top2vec_categories = {
    "Health": ["Health"],
    "Entertainment": ["Entertainment", "Arts", "Culture", "Dance", "Music", "Movies", "Actors"],
    "Environment": ["Green", "Environment", "Nature", "Climate", "Weather"],
    "Business": ["Business", "Money", "Finance", "Stocks","Trade"],
    "Politics": ["Politics", "Government", "Policies", "Election", "Strike"],
    "War": ["War","Attack"],
    "Travel": ["Travel"],
    "Technology": ["Technology", "Science"],
    "World News": ["Countries", "World"],
    "Sports": ["Sports", "Olympics"],
    "Crime": ["Crime", "murder"]
}



module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

categories_count = {}
categories_count = {category: 0 for category in dict_categories}


app = Flask(__name__)


tokenizer = AutoTokenizer.from_pretrained("dima806/news-category-classifier-distilbert")
classification_model = AutoModelForSequenceClassification.from_pretrained("dima806/news-category-classifier-distilbert")



def create_bar_chart(data):
    print(data)
    data = {category: count for category, count in data.items() if count > 0}

    labels = list(data.keys())
    values = list(data.values())

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values, text=values, textposition="auto"))

    fig.update_layout(
        title="Category Counts",
        xaxis=dict(title="Category"),
        yaxis=dict(title="Count"),
        showlegend=False,
    )

    categories_count.clear()

    plot_div = plot(fig, output_type="div")

    return plot_div

def categorize_text(text):
    for category, keywords in dict_categories.items():
        for keyword in keywords:
            if keyword.lower() in text.lower():
                return category
    return None


def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]',' ',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),' ',text)
    text = re.sub('\w*\d\w*',' ',text)
    text = re.sub(r"\\", "", text)

    return text


def perform_topic_modeling(df):

    df['text'] = df['text'].apply(clean_text)

    content = list(df['text'])

    topic_model = Top2Vec(list(content),embedding_model=model)

    categories_words = {}
    categories_words = {category: 0 for category in top2vec_categories}

    for key, words_list in top2vec_categories.items():
        temp_list = []
        
        for word in words_list:
            try:
                words, word_scores = topic_model.similar_words(keywords=[word], num_words=20)
                for word, score in zip(words, word_scores):
                    print(word)
                    if score >= 0.45:
                        temp_list.append(word)
            except ValueError as e:
                continue
        
        categories_words[key] = list(temp_list)


    for category, keywords in categories_words.items():
        documents, document_score, document_ids = topic_model.search_documents_by_keywords(keywords=keywords, num_docs=len(df))
        docm = [doc.strip().lower() for doc, score, doc_id in zip(documents, document_score, document_ids) if score > 0.01 and doc.strip().lower() != 'name']
        categories_count[category] = len(docm)

    



def perform_model_classification(data):
    for _, row in data.iterrows():
        text = row["text"]
        tokens = tokenizer(text, return_tensors="pt")
        outputs = classification_model(**tokens)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        max_prob, max_index = torch.max(probs, dim=1)
        predicted = categories[max_index.item()]
        predicted = categorize_text(predicted)
        print(predicted)
        categories_count[predicted] += 1


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/perform_action", methods=["POST"])
def perform_action():
    file = request.files["file"]
    action = request.form["action"]
    df = pd.read_csv(file)

    if action == "topic_modeling":
        perform_topic_modeling(df)
        graph = create_bar_chart(categories_count)
        return render_template(
            "result.html", message="Topic Modeling completed.", plot_div=graph
        )

    elif action == "model_classification":
        perform_model_classification(df)
        graph = create_bar_chart(categories_count)
        return render_template(
            "result.html", message="Topic Modeling completed.", plot_div=graph
        )


if __name__ == "__main__":
    app.run(debug=False)
