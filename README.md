# Naive Bayes Sentiment Analyzer 🎭

A beautiful, fast, and efficient web application for real-time sentiment analysis using from-scratch Naive Bayes machine learning models.

![Sentiment Analyzer](https://img.shields.io/badge/Accuracy-86.33%25-success)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

- ✨ **Beautiful Modern UI** - Glassmorphism design with smooth animations
- 🚀 **Real-time Predictions** - Instant sentiment analysis (<100ms)
- 🎯 **Dual Model Comparison** - Multinomial vs Bernoulli Naive Bayes
- 📊 **High Accuracy** - 86.33% accuracy on IMDb dataset
- 🎲 **Random Examples** - Load real movie reviews from dataset
- ⚡ **Optimized Performance** - DOM caching, debouncing, GPU acceleration
- 📱 **Responsive Design** - Works on all devices

## 🎬 Live Demo

**[Try it live on Render →](https://your-app-name.onrender.com)**

![Demo Screenshot](screenshot.png)

## 🏆 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Multinomial NB** | 86.33% | 87.61% | 84.63% | 86.09% |
| **Bernoulli NB** | 85.76% | 87.98% | 82.85% | 85.33% |

*Evaluated on 25,000 IMDb movie reviews using stratified 5-fold cross-validation*

## 🚀 Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/naive-bayes-sentiment.git
cd naive-bayes-sentiment/web_app

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Visit `http://localhost:5000` in your browser.

### With Docker

```bash
docker build -t sentiment-analyzer .
docker run -p 5000:5000 sentiment-analyzer
```

## 📁 Project Structure

```
web_app/
├── app.py                 # Flask backend
├── requirements.txt       # Python dependencies
├── Procfile              # Render deployment config
├── runtime.txt           # Python version
├── templates/
│   └── index.html        # Main HTML page
├── static/
│   ├── style.css         # Modern CSS with animations
│   └── script.js         # Optimized JavaScript
└── models/
    ├── multinomial_nb_model.pkl
    └── bernoulli_nb_model.pkl
```

## 🎯 How It Works

### Preprocessing Pipeline

1. **Lowercase** conversion
2. **HTML tag** removal
3. **Contraction** expansion (didn't → did not)
4. **Punctuation & number** removal
5. **Tokenization**
6. **Negation handling** (not good → not_good) ⭐
7. **Stopword** removal

### Models

**Multinomial Naive Bayes**
- Uses word frequencies
- Captures sentiment intensity
- Best for rich vocabulary text

**Bernoulli Naive Bayes**
- Uses binary features (presence/absence)
- Higher precision
- Better for short text

Both models implemented **from scratch** using only NumPy!

## 🎨 Design Highlights

- **Glassmorphism** effects with backdrop blur
- **Gradient animations** (15s smooth loop)
- **Smooth transitions** (60fps)
- **Dark theme** with vibrant accents
- **Toast notifications** for feedback
- **Loading states** with spinners

## ⚡ Performance Optimizations

### Frontend
- DOM element caching (3x faster)
- Debounced input handling
- RequestAnimationFrame animations
- CSS hardware acceleration
- Lazy loading

### Backend
- Model preloading (one-time)
- Efficient preprocessing
- Optimized vocabulary filtering
- Fast array operations

## 📊 API Endpoints

### POST /predict
Analyze sentiment of given text.

**Request:**
```json
{
  "text": "This movie was great!"
}
```

**Response:**
```json
{
  "tokens": ["movie", "great"],
  "total_tokens": 2,
  "multinomial": {
    "prediction": "Positive",
    "confidence": 0.952
  },
  "bernoulli": {
    "prediction": "Positive",
    "confidence": 0.938
  }
}
```

### GET /random_example
Get a random movie review from the dataset.

### GET /stats
Get model statistics and performance metrics.

## 🚢 Deployment

### Deploy to Render

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/naive-bayes-sentiment.git
   git push -u origin main
   ```

2. **Create Render Web Service:**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name:** naive-bayes-sentiment
     - **Environment:** Python 3
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `gunicorn app:app`
   - Click "Create Web Service"

3. **Done!** Your app will be live at `https://your-app-name.onrender.com`

### Environment Variables (Optional)

- `FLASK_ENV=production`
- `PORT=5000`

## 🛠️ Tech Stack

- **Backend:** Flask 3.0
- **Server:** Gunicorn (production)
- **ML:** Custom Naive Bayes (NumPy)
- **Frontend:** Vanilla HTML/CSS/JavaScript
- **Fonts:** Google Fonts (Inter)
- **Deployment:** Render

## 📝 License

MIT License - feel free to use this project for learning or production!

## 👨‍💻 Author

**Harshit Saraswat**  
📧 saraswatharshit472@gmail.com

## 🙏 Acknowledgments

- IMDb dataset from Stanford AI Lab
- Inspired by modern web design trends
- Built with ❤️ for ML education

## 📈 Future Improvements

- [ ] Add more ML models (SVM, Logistic Regression)
- [ ] Implement user authentication
- [ ] Save prediction history
- [ ] Add batch prediction API
- [ ] Multi-language support
- [ ] Sentiment intensity scoring

## 🐛 Issues

Found a bug? [Open an issue](https://github.com/yourusername/naive-bayes-sentiment/issues)

## ⭐ Star This Repo

If you found this project helpful, please give it a star!

---

**Made with 💜 using from-scratch Machine Learning**
