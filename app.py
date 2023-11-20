from flask import Flask, render_template, request, redirect
import pandas as pd
import plotly.express as px
from plotly.offline import plot
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import plotly.graph_objects as go


categories = [
    "ARTS",
    "ARTS & CULTURE",
    "BLACK VOICES",
    "BUSINESS",
    "COLLEGE",
    "COMEDY",
    "CRIME",
    "CULTURE & ARTS",
    "DIVORCE",
    "EDUCATION",
    "ENTERTAINMENT",
    "ENVIRONMENT",
    "FIFTY",
    "FOOD & DRINK",
    "GOOD NEWS",
    "GREEN",
    "HEALTHY LIVING",
    "HOME & LIVING",
    "IMPACT",
    "LATINO VOICES",
    "MEDIA",
    "MONEY",
    "PARENTING",
    "PARENTS",
    "POLITICS",
    "QUEER VOICES",
    "RELIGION",
    "SCIENCE",
    "SPORTS",
    "STYLE",
    "STYLE & BEAUTY",
    "TASTE",
    "TECH",
    "THE WORLDPOST",
    "TRAVEL",
    "U.S. NEWS",
    "WEDDINGS",
    "WEIRD NEWS",
    "WELLNESS",
    "WOMEN",
    "WORLD NEWS",
    "WORLDPOST",
]


dict_categories = {
    "Arts": ["ARTS & CULTURE", "CULTURE & ARTS", "ARTS", "QUEER VOICES"],
    "Health": ["WELLNESS", "HEALTHY LIVING"],
    "Entertainment": ["ENTERTAINMENT", "MEDIA", "COMEDY", "WEDDINGS"],
    "Home": ["WOMEN", "PARENTS", "PARENTING", "HOME & LIVING"],
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


categories_count = {}
categories_count = {category: 0 for category in dict_categories}


app = Flask(__name__)


tokenizer = AutoTokenizer.from_pretrained("dima806/news-category-classifier-distilbert")
model = AutoModelForSequenceClassification.from_pretrained(
    "dima806/news-category-classifier-distilbert"
)


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


def perform_topic_modeling(data):
    # Add your topic modeling logic here
    pass


def perform_model_classification(data):
    for _, row in data.iterrows():
        text = row["text"]
        tokens = tokenizer(text, return_tensors="pt")
        outputs = model(**tokens)
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
        perform_model_classification(df)
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
    app.run(debug=True)
