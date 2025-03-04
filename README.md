# Klasifikasi Kopi

Proyek ini mengimplementasikan model deep learning untuk klasifikasi biji kopi menggunakan PyTorch.

## Persiapan

1. Instal dependensi:
```bash
pip install -r requirements.txt
```

2. Siapkan dataset dengan struktur berikut:
```
dataset_kopi/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
├── class2/
│   ├── image3.jpg
│   ├── image4.jpg
...
```

## Training

Jalankan pelatihan model dengan perintah berikut:

> Jangan lupa untuk mengganti nama folder dataset menjadi `dataset_kopi`

```bash
python train.py --model efficientnet --batch_size 32 --lr 0.001 --epochs 30
```

### Model yang Tersedia

- `efficientnet`: EfficientNet B0
- `shufflenet`: ShuffleNet V2 X0.5
- `resnet152`: ResNet 152
- `vit`: Vision Transformer B/16
- `all`: Melatih semua model secara berurutan

### Parameter Command-line

```
--model MODEL         Arsitektur model yang digunakan ('all' untuk menjalankan semua model)
--batch_size SIZE     Batch size untuk training (default: 32)
--lr RATE             Learning rate (default: 0.001)
--epochs NUM          Jumlah epoch (default: 50)
--patience NUM        Batas kesabaran untuk early stopping (default: 5)
--device DEVICE       Device yang digunakan: cuda, mps, atau cpu (mengganti deteksi otomatis)
--no-wandb            Menonaktifkan logging Weights & Biases
--folds NUM           Jumlah fold untuk cross-validation (default: 5)
--fastmode            Mengaktifkan mode cepat untuk pengujian
```

### K-Fold Cross-Validation

Proses training menggunakan k-fold cross-validation secara default. Jumlah fold dapat diatur dengan parameter `--folds`.

Contoh:
```bash
python train.py --model efficientnet --folds 3
```

### Fast Mode

Untuk pengujian cepat atau pengembangan, Anda dapat mengaktifkan fast mode:

```bash
python train.py --model efficientnet --fastmode
```

Ini akan:
- Menggunakan hanya sebagian kecil gambar per kelas
- Menjalankan lebih sedikit epoch
- Menggunakan lebih sedikit fold untuk validasi

## Integrasi Weights & Biases

Proyek ini menggunakan Weights & Biases untuk pelacakan eksperimen. Setiap run secara otomatis diberi nama menggunakan format timestamp `model_YYYYMMDD-HHMM`.

Untuk menonaktifkan logging wandb, gunakan flag `--no-wandb`:

```bash
python train.py --model efficientnet --no-wandb
```

Untuk melihat hasil eksperimen, kunjungi dashboard Weights & Biases di [wandb.ai](https://wandb.ai).

Hasil cross-validation secara otomatis dicatat, termasuk:
- Metrik training dan validation per fold
- Rata-rata dan standar deviasi akurasi di seluruh fold
- Plot history training
