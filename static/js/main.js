// static/js/main.js
window.addEventListener('load', () => {
    const bodyElement = document.body;
    const canvas = document.getElementById('drawing-canvas');
    const canvasWrapper = document.querySelector('.canvas-wrapper');
    const resultContainer = document.getElementById('result-container');
    const colorChartSidebar = document.getElementById('color-chart-sidebar');
    const clearBtn = document.getElementById('clear-btn');
    const ctx = canvas.getContext('2d');

    let history = [];
    let drawing = false;
    let lastX = 0;
    let lastY = 0;

    const updateClearButtonState = () => {
        if (history.length > 1) clearBtn.classList.add('clear-btn-active');
        else clearBtn.classList.remove('clear-btn-active');
    };

    const clearAndReset = () => {
        bodyElement.classList.remove('confidence-low', 'confidence-mid', 'confidence-high', 'confidence-very-high');
        colorChartSidebar.classList.remove('is-visible');
        canvasWrapper.classList.add('is-clearing');
        
        setTimeout(() => {
            ctx.fillStyle = "white";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            history = [];
            saveState();
            resultContainer.classList.remove('has-result');
            document.getElementById('prediction-result').textContent = '-';
            document.getElementById('confidence-result').textContent = '-';
            updateClearButtonState();
            canvasWrapper.classList.remove('is-clearing');
        }, 500);
    };

    const saveState = () => history.push(canvas.toDataURL());

    const undoLast = () => {
        if (history.length <= 1) {
            clearAndReset();
            return;
        }
        history.pop();
        const lastStateUrl = history[history.length - 1];
        const img = new Image();
        img.src = lastStateUrl;
        img.onload = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
        };
        updateClearButtonState();
    };

    const startDrawing = (e) => {
        drawing = true;
        [lastX, lastY] = [e.offsetX, e.offsetY];
    };

    const draw = (e) => {
        if (!drawing) return;
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        [lastX, lastY] = [e.offsetX, e.offsetY];
    };

    const stopDrawing = () => {
        if (!drawing) return;
        drawing = false;
        saveState();
        updateClearButtonState();
    };

    const predictCharacter = () => {
        if (history.length <= 1) {
            alert("Please draw a character before predicting!");
            return;
        }
        const imageData = canvas.toDataURL('image/png');
        
        const predictionResultElem = document.getElementById('prediction-result');
        const confidenceResultElem = document.getElementById('confidence-result');

        resultContainer.classList.remove('has-result');
        predictionResultElem.textContent = '...';
        confidenceResultElem.textContent = '';
        canvasWrapper.classList.add('is-predicting');
        bodyElement.classList.remove('confidence-low', 'confidence-mid', 'confidence-high', 'confidence-very-high');

        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ imageData: imageData })
        })
        .then(response => response.json())
        .then(data => {
            if (data.prediction) {
                const confidence = data.confidence;
                
                if (confidence > 0.9) bodyElement.classList.add('confidence-very-high');
                else if (confidence > 0.7) bodyElement.classList.add('confidence-high');
                else if (confidence > 0.5) bodyElement.classList.add('confidence-mid');
                else bodyElement.classList.add('confidence-low');

                predictionResultElem.textContent = data.prediction;
                confidenceResultElem.textContent = `${(confidence * 100).toFixed(2)}%`;
                resultContainer.classList.add('has-result');
                colorChartSidebar.classList.add('is-visible');
            }
        })
        .catch(error => console.error('Error:', error))
        .finally(() => canvasWrapper.classList.remove('is-predicting'));
    };

    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';
    clearAndReset();

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    document.getElementById('clear-btn').addEventListener('click', clearAndReset);
    document.getElementById('predict-btn').addEventListener('click', predictCharacter);

    window.addEventListener('keydown', (e) => {
        const key = e.key.toLowerCase();
        
        if (e.ctrlKey || e.metaKey) {
            if (key === 'z') {
                e.preventDefault();
                undoLast();
            }
            return;
        }

        if (key === 'p') {
            e.preventDefault();
            predictCharacter();
        } else if (key === 'c') {
            e.preventDefault();
            clearAndReset();
        }
    });
});