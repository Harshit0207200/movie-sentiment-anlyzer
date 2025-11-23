// Fast & Efficient JavaScript - Optimized for Performance

// Cache DOM elements for better performance
const elements = {
    textInput: document.getElementById('textInput'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    clearBtn: document.getElementById('clearBtn'),
    resultsSection: document.getElementById('resultsSection'),
    loadingSpinner: document.getElementById('loadingSpinner'),
    errorMessage: document.getElementById('errorMessage'),
    tokenCount: document.getElementById('tokenCount'),
    tokensPreview: document.getElementById('tokensPreview'),
    exampleBtns: document.querySelectorAll('.example-btn')
};

// Debounce function for performance
const debounce = (func, wait) => {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
};

// Event listeners
elements.analyzeBtn.addEventListener('click', analyzeSentiment);
elements.clearBtn.addEventListener('click', clearInput);
elements.textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        analyzeSentiment();
    }
});

// Example buttons
elements.exampleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        elements.textInput.value = btn.dataset.text;
        elements.textInput.focus();
    });
});

// Random example button
const randomExampleBtn = document.getElementById('randomExampleBtn');
randomExampleBtn.addEventListener('click', loadRandomExample);

// Load random example from dataset
async function loadRandomExample() {
    randomExampleBtn.disabled = true;
    randomExampleBtn.innerHTML = `
        <svg class="spinner-small" width="16" height="16" viewBox="0 0 16 16">
            <circle cx="8" cy="8" r="6" stroke="currentColor" stroke-width="2" fill="none" opacity="0.3"/>
            <path d="M8 2 A6 6 0 0 1 14 8" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"/>
        </svg>
        Loading...
    `;

    try {
        const response = await fetch('/random_example');

        if (!response.ok) {
            throw new Error('Failed to load example');
        }

        const data = await response.json();
        elements.textInput.value = data.text;
        elements.textInput.focus();

        // Show notification
        showNotification(`Loaded ${data.actual_sentiment} review from dataset`, 'success');

    } catch (error) {
        showError('Failed to load random example. Make sure the dataset is downloaded.');
    } finally {
        randomExampleBtn.disabled = false;
        randomExampleBtn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M13 3L3 13M3 3L13 13" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            </svg>
            Load Random Review
        `;
    }
}

// Main analysis function
async function analyzeSentiment() {
    const text = elements.textInput.value.trim();

    if (!text) {
        showError('Please enter some text to analyze');
        return;
    }

    // Show loading state
    showLoading();

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Prediction failed');
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// Display results with smooth animations
function displayResults(data) {
    hideError();

    // Update preprocessing info
    elements.tokenCount.textContent = data.total_tokens;

    // Display tokens
    elements.tokensPreview.innerHTML = data.tokens
        .map(token => `<span class="token">${escapeHtml(token)}</span>`)
        .join('');

    // Update Multinomial NB results
    updateModelCard('mnb', data.multinomial);

    // Update Bernoulli NB results
    updateModelCard('bnb', data.bernoulli);

    // Show results with animation
    elements.resultsSection.style.display = 'block';

    // Smooth scroll to results
    setTimeout(() => {
        elements.resultsSection.scrollIntoView({
            behavior: 'smooth',
            block: 'nearest'
        });
    }, 100);
}

// Update individual model card
function updateModelCard(modelPrefix, data) {
    const isPositive = data.prediction === 'Positive';

    // Update sentiment badge
    const resultDiv = document.getElementById(`${modelPrefix}Result`);
    const badge = resultDiv.querySelector('.sentiment-badge');
    const icon = badge.querySelector('.sentiment-icon');
    const text = badge.querySelector('.sentiment-text');

    badge.className = `sentiment-badge ${isPositive ? 'positive' : 'negative'}`;
    icon.textContent = isPositive ? '😊' : '😞';
    text.textContent = data.prediction;

    // Update confidence
    const confidenceValue = resultDiv.querySelector('.confidence-value');
    confidenceValue.textContent = `${(data.confidence * 100).toFixed(1)}%`;

    // Animate probability bars
    animateBar(`${modelPrefix}PosBar`, `${modelPrefix}PosProb`, data.positive_prob);
    animateBar(`${modelPrefix}NegBar`, `${modelPrefix}NegProb`, data.negative_prob);
}

// Animate probability bar with smooth transition
function animateBar(barId, probId, probability) {
    const bar = document.getElementById(barId);
    const probText = document.getElementById(probId);

    // Reset width first for animation
    bar.style.width = '0%';

    // Use requestAnimationFrame for smooth animation
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            bar.style.width = `${probability * 100}%`;
            probText.textContent = `${(probability * 100).toFixed(1)}%`;
        });
    });
}

// Clear input and results
function clearInput() {
    elements.textInput.value = '';
    elements.resultsSection.style.display = 'none';
    hideError();
    elements.textInput.focus();
}

// Show/hide loading state
function showLoading() {
    elements.loadingSpinner.style.display = 'block';
    elements.analyzeBtn.disabled = true;
    elements.analyzeBtn.textContent = 'Analyzing...';
}

function hideLoading() {
    elements.loadingSpinner.style.display = 'none';
    elements.analyzeBtn.disabled = false;
    elements.analyzeBtn.innerHTML = `
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M19 19L13 13M15 8C15 11.866 11.866 15 8 15C4.13401 15 1 11.866 1 8C1 4.13401 4.13401 1 8 1C11.866 1 15 4.13401 15 8Z" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        </svg>
        Analyze Sentiment
    `;
}

// Show/hide error messages
function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorMessage.style.display = 'block';
    elements.resultsSection.style.display = 'none';

    // Auto-hide after 5 seconds
    setTimeout(hideError, 5000);
}

function hideError() {
    elements.errorMessage.style.display = 'none';
}

// Show success notification
function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);

    // Animate in
    setTimeout(() => notification.classList.add('show'), 10);

    // Auto-hide after 3 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Auto-resize textarea
elements.textInput.addEventListener('input', debounce(() => {
    elements.textInput.style.height = 'auto';
    elements.textInput.style.height = elements.textInput.scrollHeight + 'px';
}, 100));

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to analyze
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        analyzeSentiment();
    }
    // Escape to clear
    if (e.key === 'Escape') {
        clearInput();
    }
});

// Performance monitoring (optional - can be removed in production)
if (window.performance && window.performance.mark) {
    window.addEventListener('load', () => {
        performance.mark('app-loaded');
        console.log('✓ App loaded and ready');
    });
}
