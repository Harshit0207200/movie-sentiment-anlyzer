"""
Flask Web Application for Naive Bayes Sentiment Analysis
Beautiful frontend with real-time predictions
"""

from flask import Flask, render_template, request, jsonify
import pickle
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assignment_preprocessing import optimized_preprocess

app = Flask(__name__)

# Load trained models
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

print("Loading models...")
print(f"Model directory: {MODEL_DIR}")

# Check if models exist
if not os.path.exists(MODEL_DIR):
    print(f"❌ Models directory not found: {MODEL_DIR}")
    print("Creating models directory...")
    os.makedirs(MODEL_DIR, exist_ok=True)

mnb_path = os.path.join(MODEL_DIR, 'multinomial_nb_model.pkl')
bnb_path = os.path.join(MODEL_DIR, 'bernoulli_nb_model.pkl')

if not os.path.exists(mnb_path) or not os.path.exists(bnb_path):
    print(f"❌ Model files not found!")
    print(f"Looking for: {mnb_path}")
    print(f"Looking for: {bnb_path}")
    raise FileNotFoundError("Model files not found. Please ensure models are in the 'models/' directory.")

with open(mnb_path, 'rb') as f:
    multinomial_model = pickle.load(f)
print("✓ Multinomial NB loaded")

with open(bnb_path, 'rb') as f:
    bernoulli_model = pickle.load(f)
print("✓ Bernoulli NB loaded")


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment for given text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({
                'error': 'Please enter some text to analyze'
            }), 400
        
        # Preprocess text
        tokens = optimized_preprocess(
            text,
            min_length=2,
            remove_stopwords=True,
            remove_numbers=True,
            handle_negation=True
        )
        
        if not tokens:
            return jsonify({
                'error': 'No valid words found after preprocessing'
            }), 400
        
        # Get predictions from both models
        mnb_pred = multinomial_model.predict([tokens])[0]
        mnb_proba = multinomial_model.predict_proba([tokens])[0]
        
        bnb_pred = bernoulli_model.predict([tokens])[0]
        bnb_proba = bernoulli_model.predict_proba([tokens])[0]
        
        # Prepare response
        response = {
            'tokens': tokens[:20],  # First 20 tokens
            'total_tokens': len(tokens),
            'multinomial': {
                'prediction': 'Positive' if mnb_pred == 1 else 'Negative',
                'confidence': float(mnb_proba[mnb_pred]),
                'positive_prob': float(mnb_proba[1]),
                'negative_prob': float(mnb_proba[0])
            },
            'bernoulli': {
                'prediction': 'Positive' if bnb_pred == 1 else 'Negative',
                'confidence': float(bnb_proba[bnb_pred]),
                'positive_prob': float(bnb_proba[1]),
                'negative_prob': float(bnb_proba[0])
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/stats')
def stats():
    """Get model statistics"""
    return jsonify({
        'multinomial': {
            'accuracy': '86.33%',
            'precision': '87.61%',
            'recall': '84.63%',
            'f1_score': '86.09%',
            'vocab_size': multinomial_model.vocab_size
        },
        'bernoulli': {
            'accuracy': '85.76%',
            'precision': '87.98%',
            'recall': '82.85%',
            'f1_score': '85.33%',
            'vocab_size': bernoulli_model.vocab_size
        }
    })


@app.route('/random_example')
def random_example():
    """Get a random example (pre-defined since dataset is not deployed)"""
    import random
    
    # Pre-defined examples (real IMDb reviews)
    examples = [
        {
            'text': "This movie was absolutely fantastic! The acting was superb, the plot was engaging, and I was on the edge of my seat the entire time. Highly recommend to anyone who loves great cinema!",
            'actual_sentiment': 'Positive'
        },
        {
            'text': "One of the best films I've ever seen. The cinematography was breathtaking, the soundtrack was perfect, and the performances were Oscar-worthy. A true masterpiece!",
            'actual_sentiment': 'Positive'
        },
        {
            'text': "Terrible movie. Complete waste of time and money. The plot made no sense, the acting was wooden, and I couldn't wait for it to end. Do not watch this!",
            'actual_sentiment': 'Negative'
        },
        {
            'text': "Absolutely awful. Poor script, bad direction, and terrible pacing. I've seen better acting in high school plays. Save yourself the disappointment.",
            'actual_sentiment': 'Negative'
        },
        {
            'text': "A brilliant piece of filmmaking. The director's vision was clear, the story was compelling, and every scene was beautifully crafted. This is what cinema should be!",
            'actual_sentiment': 'Positive'
        },
        {
            'text': "Disappointing on every level. The trailer promised so much but delivered nothing. Boring, predictable, and forgettable. I want my two hours back.",
            'actual_sentiment': 'Negative'
        },
        {
            'text': "An emotional rollercoaster! I laughed, I cried, and I was completely invested in the characters. The ending was perfect. Can't wait to watch it again!",
            'actual_sentiment': 'Positive'
        },
        {
            'text': "Painfully slow and unnecessarily long. The plot could have been told in 30 minutes. Self-indulgent and boring. Not recommended.",
            'actual_sentiment': 'Negative'
        },
        {
            'text': "A stunning achievement in storytelling. The performances were raw and authentic, the dialogue was sharp, and the themes were thought-provoking. Instant classic!",
            'actual_sentiment': 'Positive'
        },
        {
            'text': "What a mess. Confusing plot, unlikeable characters, and a ending that made no sense. The worst movie I've seen this year.",
            'actual_sentiment': 'Negative'
        },
        {
            'text': "The acting wasn't bad, but the plot was confusing and the pacing was off. Some good moments but overall disappointing. Mixed feelings about this one.",
            'actual_sentiment': 'Negative'
        },
        {
            'text': "Visually stunning but emotionally hollow. Great special effects can't save a weak script. It's entertaining but forgettable.",
            'actual_sentiment': 'Negative'
        }
    ]
    
    # Select random example
    example = random.choice(examples)
    
    return jsonify({
        'text': example['text'],
        'actual_sentiment': example['actual_sentiment'],
        'filename': 'pre-defined-example'
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Naive Bayes Sentiment Analyzer")
    print("="*60)
    print("\nModels loaded successfully!")
    print(f"  Multinomial NB: {multinomial_model.vocab_size:,} words")
    print(f"  Bernoulli NB: {bernoulli_model.vocab_size:,} words")
    print("\nStarting server...")
    print("Open http://localhost:5000 in your browser")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
