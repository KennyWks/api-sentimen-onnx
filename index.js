import express from 'express';
import { runModel } from './api/sentimen.js';
const app = express();
const PORT = 3000;

// 1. WAJIB: Gunakan middleware ini agar bisa membaca req.body
app.use(express.json());

app.get('/', (req, res) => {
    res.send('Selamat datang di API Sentimen!');
});

// 2. Buat POST endpoint untuk menerima input teks
app.post('/sentiment', async (req, res) => {
    try {
        // Mengambil teks dari body request
        const { text } = req.body;

        // Validasi jika input teks kosong
        if (!text) {
            return res.status(400).json({
                success: false,
                message: 'Teks tidak boleh kosong!'
            });
        }

        const result = await runModel(text);
        console.log('Hasil prediksi:', result);
        return res.status(200).json({
            success: true,
            input: text,
            result: {
                label: result.label,
                confidence: result.confidence
            },
        });

    } catch (error) {
        console.error(error);
        return res.status(500).json({
            success: false,
            message: 'Terjadi kesalahan pada server.'
        });
    }
});

app.listen(PORT, () => {
    console.log(`Server berjalan di http://localhost:${PORT}`);
});