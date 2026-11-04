import fs from 'node:fs';
import https from 'node:https';
import path from 'node:path';
import * as ort from 'onnxruntime-node';
import { AutoTokenizer, env } from '@xenova/transformers';

// --- KONFIGURASI HUGGING FACE & VERCEL ---
// Ganti dengan username dan nama repositori Hugging Face Anda
const REPO_ID = 'username/sentimen-model-onnx'; 
const MODEL_URL = `https://huggingface.co/${REPO_ID}/resolve/main/model.onnx`;
const DATA_URL = `https://huggingface.co/${REPO_ID}/resolve/main/model.onnx.data`;

const TMP_MODEL_PATH = '/tmp/model.onnx';
const TMP_DATA_PATH = '/tmp/model.onnx.data';

// Konfigurasi Tokenizer (@xenova) agar mengambil dari Remote dan menyimpan di /tmp/
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.cacheDir = '/tmp/';

// Fungsi bantuan untuk mengunduh file ONNX jika belum ada di /tmp/
function downloadFile(url, dest) {
    return new Promise((resolve, reject) => {
        // Jika file sudah ada di instance serverless, lewati pengunduhan
        if (fs.existsSync(dest)) {
            return resolve();
        }
        
        const file = fs.createWriteStream(dest);
        https.get(url, (response) => {
            if (response.statusCode === 302 || response.statusCode === 301) {
                https.get(response.headers.location, (res) => {
                    res.pipe(file);
                    file.on('finish', () => file.close(resolve));
                }).on('error', reject);
            } else {
                response.pipe(file);
                file.on('finish', () => file.close(resolve));
            }
        }).on('error', (err) => {
            fs.unlink(dest, () => {});
            reject(err);
        });
    });
}

export async function runModel(text) {
    if (typeof text !== 'string' || !text.trim()) {
        throw new TypeError('Teks tidak boleh kosong.');
    }

    // 1. Pastikan model.onnx dan model.onnx.data sudah terunduh di /tmp/ Vercel
    await Promise.all([
        downloadFile(MODEL_URL, TMP_MODEL_PATH),
        downloadFile(DATA_URL, TMP_DATA_PATH)
    ]);

    // 2. Load Tokenizer dari Hugging Face dan Load Model dari /tmp/
    const [tokenizer, session] = await Promise.all([
        AutoTokenizer.from_pretrained(REPO_ID),
        ort.InferenceSession.create(TMP_MODEL_PATH),
    ]);

    // 3. Proses Tokenisasi
    const tokens = await tokenizer(text, {
        padding: 'max_length',
        truncation: true,
        max_length: 128,
    });
    
    // 4. Siapkan Input untuk Model ONNX
    const feeds = Object.fromEntries(
        session.inputNames.map((name) => {
            const tensor = tokens[name];
            if (!tensor) {
                throw new Error(`Input model "${name}" tidak dihasilkan oleh tokenizer.`);
            }
            return [name, new ort.Tensor(tensor.type, tensor.data, tensor.dims)];
        }),
    );

    // 5. Jalankan Inferensi
    const results = await session.run(feeds);
    
    // 6. Hitung Probabilitas menggunakan Softmax
    const probabilities = softmax(results[session.outputNames[0]].data);
    
    // 7. Cari indeks dengan nilai tertinggi (Argmax)
    const maxIndex = probabilities.indexOf(Math.max(...probabilities));
    
    // 8. Petakan indeks ke label
    const labels = ["Positif", "Netral", "Negatif"];
    
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