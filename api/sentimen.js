import path from 'node:path';
import * as ort from 'onnxruntime-node';
import { AutoTokenizer, env } from '@xenova/transformers';

export async function runModel(text) {
    if (typeof text !== 'string' || !text.trim()) {
        throw new TypeError('Teks tidak boleh kosong.');
    }

    const modelDirectory = path.resolve(import.meta.dirname, '..', 'model');
    env.localModelPath = path.dirname(modelDirectory);
    env.allowRemoteModels = false;

    const [tokenizer, session] = await Promise.all([
        AutoTokenizer.from_pretrained('model'),
        ort.InferenceSession.create(path.join(modelDirectory, 'model.onnx')),
    ]);

    const tokens = await tokenizer(text, {
        padding: 'max_length',
        truncation: true,
        max_length: 128,
    });
    const feeds = Object.fromEntries(
        session.inputNames.map((name) => {
            const tensor = tokens[name];
            if (!tensor) {
                throw new Error(`Input model "${name}" tidak dihasilkan oleh tokenizer.`);
            }
            return [name, new ort.Tensor(tensor.type, tensor.data, tensor.dims)];
        }),
    );

    const results = await session.run(feeds);
    
    // 2. Hitung Probabilitas menggunakan Softmax
    const probabilities = softmax(results[session.outputNames[0]].data);
    // console.log("Probabilitas:", probabilities); 
    
    // 3. Cari indeks dengan nilai tertinggi (Argmax)
    const maxIndex = probabilities.indexOf(Math.max(...probabilities));
    
    // 4. Petakan indeks ke label
    const labels = ["Positif", "Netral", "Negatif"];
    // console.log(`Hasil Prediksi: ${labels[maxIndex]} dengan tingkat keyakinan ${(probabilities[maxIndex] * 100).toFixed(2)}%`);
    
    return {
        label: labels[maxIndex],
        confidence: (probabilities[maxIndex] * 100).toFixed(2) + '%'
    }
}

function softmax(logits) {
    const maxLogit = Math.max(...logits); // Untuk stabilitas numerik
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const sumScores = scores.reduce((a, b) => a + b, 0);
    return scores.map(s => s / sumScores);
}
