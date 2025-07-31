document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const status = document.getElementById('status');
    const modelSelector = document.getElementById('model-selector');
    const ctx = canvas.getContext('2d');
    const focusControl = document.getElementById('focus-control');
    const focusSlider = document.getElementById('focus-slider');

    let session;
    let labels = [];
    let detectionLoopId = null;

    async function loadModel() {
        if (detectionLoopId) {
            cancelAnimationFrame(detectionLoopId);
            detectionLoopId = null;
        }
        const selectedOption = modelSelector.options[modelSelector.selectedIndex];
        const modelFolder = selectedOption.value;
        const modelFile = selectedOption.dataset.filename;
        const modelPath = `./modelos/${modelFolder}/${modelFile}`;
        const labelsPath = `./modelos/${modelFolder}/labels.json`;

        status.textContent = `Cargando modelo: ${modelFolder}...`;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        try {
            session = await ort.InferenceSession.create(modelPath);
            const response = await fetch(labelsPath);
            labels = await response.json();
            status.textContent = 'Modelo cargado. Iniciando cámara...';
            startDetection();
        } catch (e) {
            status.textContent = `Error al cargar el modelo: ${e.message}`;
        }
    }

    modelSelector.addEventListener('change', loadModel);

    async function initCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;

            const [track] = stream.getVideoTracks();
            const capabilities = track.getCapabilities();

            if (capabilities.focusDistance) {
                focusControl.style.display = 'block'; 

                focusSlider.min = capabilities.focusDistance.min;
                focusSlider.max = capabilities.focusDistance.max;
                focusSlider.step = capabilities.focusDistance.step;
                
                focusSlider.addEventListener('input', (event) => {
                    track.applyConstraints({
                        advanced: [{
                            focusMode: 'manual',
                            focusDistance: event.target.value
                        }]
                    });
                });
            } else {
                console.log("El control manual de enfoque no es soportado por esta cámara/navegador.");
            }

            video.onloadedmetadata = () => {
                loadModel();
            };
        } catch (e) {
            status.textContent = `Error al acceder a la cámara: ${e.message}`;
        }
    }

    function startDetection() {
        status.textContent = 'Detectando objetos en tiempo real...';
        async function detectFrame() {
            const inputTensor = await preprocess(video);
            const feeds = { 'images': inputTensor };
            const results = await session.run(feeds);
            processAndDraw(results, ctx, video.videoWidth, video.videoHeight);
            detectionLoopId = requestAnimationFrame(detectFrame);
        }
        detectFrame();
    }

    async function preprocess(videoElement) {
        const modelWidth = 1280;
        const modelHeight = 1280;
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = modelWidth;
        tempCanvas.height = modelHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(videoElement, 0, 0, modelWidth, modelHeight);
        const imageData = tempCtx.getImageData(0, 0, modelWidth, modelHeight);
        const { data } = imageData;
        const float32Data = new Float32Array(3 * modelWidth * modelHeight);
        for (let i = 0; i < modelWidth * modelHeight; ++i) {
            const r = data[i * 4] / 255.0;
            const g = data[i * 4 + 1] / 255.0;
            const b = data[i * 4 + 2] / 255.0;
            float32Data[i] = r;
            float32Data[i + modelWidth * modelHeight] = g;
            float32Data[i + 2 * modelWidth * modelHeight] = b;
        }
        return new ort.Tensor('float32', float32Data, [1, 3, modelHeight, modelWidth]);
    }

    function processAndDraw(results, ctx, videoWidth, videoHeight) {
        const modelWidth = 1280;
        const modelHeight = 1280;
        const confidenceThreshold = 0.5;
        ctx.clearRect(0, 0, videoWidth, videoHeight);
        const output = results.output0.data;
        const numPredictions = results.output0.dims[2];
        const numClasses = labels.length;
        for (let i = 0; i < numPredictions; i++) {
            let maxProb = 0;
            let classId = -1;
            for (let j = 0; j < numClasses; j++) {
                const prob = output[(4 + j) * numPredictions + i];
                if (prob > maxProb) {
                    maxProb = prob;
                    classId = j;
                }
            }
            if (maxProb > confidenceThreshold) {
                const label = labels[classId] || `Clase ${classId}`;
                const x_center = output[0 * numPredictions + i];
                const y_center = output[1 * numPredictions + i];
                const width = output[2 * numPredictions + i];
                const height = output[3 * numPredictions + i];
                const x = (x_center - width / 2) / modelWidth * videoWidth;
                const y = (y_center - height / 2) / modelHeight * videoHeight;
                const w = width / modelWidth * videoWidth;
                const h = height / modelHeight * videoHeight;
                ctx.strokeStyle = '#00FF00';
                ctx.lineWidth = 3;
                ctx.strokeRect(x, y, w, h);
                ctx.fillStyle = '#00FF00';
                ctx.font = '18px sans-serif';
                ctx.fillText(`${label} (${(maxProb * 100).toFixed(1)}%)`, x, y > 10 ? y - 5 : 20);
            }
        }
    }

    initCamera();
});