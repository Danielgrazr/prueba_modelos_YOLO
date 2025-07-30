document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const status = document.getElementById('status');
    const modelSelector = document.getElementById('model-selector');
    const ctx = canvas.getContext('2d');

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

    // --- FUNCIÓN CORREGIDA ---
    async function preprocess(videoElement) {
        const modelWidth = 1280; // <-- CORREGIDO: Tamaño esperado por el modelo
        const modelHeight = 1280; // <-- CORREGIDO: Tamaño esperado por el modelo

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

    // --- FUNCIÓN CORREGIDA ---
    function processAndDraw(results, ctx, videoWidth, videoHeight) {
        const modelWidth = 1280; // <-- CORREGIDO: Tamaño usado para la predicción
        const modelHeight = 1280; // <-- CORREGIDO: Tamaño usado para la predicción

        const output = results.output0.data;
        const confidenceThreshold = 0.5;
        ctx.clearRect(0, 0, videoWidth, videoHeight);
        const stride = 4 + labels.length;
        for (let i = 0; i < output.length; i += stride) {
            const [x_center, y_center, width, height, ...class_probs] = output.slice(i, i + stride);
            let max_prob = 0;
            let class_id = -1;
            for (let j = 0; j < class_probs.length; j++) {
                if (class_probs[j] > max_prob) {
                    max_prob = class_probs[j];
                    class_id = j;
                }
            }
            if (max_prob > confidenceThreshold) {
                const label = labels[class_id] || `Clase ${class_id}`;
                
                // CORREGIDO: La escala de las cajas ahora usa el tamaño del modelo
                const x = (x_center - width / 2) / modelWidth * videoWidth;
                const y = (y_center - height / 2) / modelHeight * videoHeight;
                const w = width / modelWidth * videoWidth;
                const h = height / modelHeight * videoHeight;

                ctx.strokeStyle = '#00FF00';
                ctx.lineWidth = 3;
                ctx.strokeRect(x, y, w, h);
                ctx.fillStyle = '#00FF00';
                ctx.font = '18px sans-serif';
                ctx.fillText(`${label} (${(max_prob * 100).toFixed(1)}%)`, x, y > 10 ? y - 5 : 20);
            }
        }
    }

    initCamera();
});