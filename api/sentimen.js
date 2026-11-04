const ort = require('onnxruntime-node');
const path = require('path');

// Variabel global untuk menyimpan sesi model agar tidak dimuat ulang 
// pada setiap permintaan (mengoptimalkan waktu respons/cold start)
let session = null;

module.exports = async (req, res) => {
  // Hanya menerima metode POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Metode tidak diizinkan. Gunakan POST.' });
  }

  try {
    // 1. Muat Model (Hanya jika belum dimuat)
    if (!session) {
      // process.cwd() menunjuk ke root direktori proyek saat dijalankan di Vercel
      const modelPath = path.resolve(process.cwd(), 'model', 'model.onnx');
      session = await ort.InferenceSession.create(modelPath);
      console.log('Model ONNX berhasil dimuat.');
    }

    // 2. Ambil input dari body request (contoh: teks yang sudah diubah ke array token/vektor)
    // Format JSON yang diharapkan: { "input_data": [0.5, 0.2, 0.1, ...] }
    const { input_data } = req.body;

    if (!input_data || !Array.isArray(input_data)) {
      return res.status(400).json({ error: 'Format input tidak valid. Pastikan mengirim array.' });
    }

    // 3. Konversi input menjadi Tensor ONNX
    // Catatan: Tipe data ('float32', 'int64') dan dimensi (shape) HARUS sesuai 
    // dengan spesifikasi model ONNX Anda saat dilatih.
    // Contoh di bawah membuat tensor 2D dengan bentuk [1, jumlah_input]
    const tensor = new ort.Tensor(
      'float32', 
      new Float32Array(input_data), 
      [1, input_data.length] 
    );

    // 4. Siapkan feed (sesuaikan kunci dengan nama input layer model Anda)
    const inputName = session.inputNames[0];
    const feeds = { [inputName]: tensor };

    // 5. Jalankan inferensi
    const results = await session.run(feeds);

    // 6. Ambil output
    const outputName = session.outputNames[0];
    const outputData = results[outputName].data;

    // Kembalikan respons dalam format JSON (Array.from mengubah Float32Array kembali ke array standar)
    return res.status(200).json({
      status: 'sukses',
      prediksi: Array.from(outputData)
    });

  } catch (error) {
    console.error('Terjadi kesalahan inferensi:', error);
    return res.status(500).json({ error: error.message });
  }
};