# CSC4005 – Lab 2 Report

## 1. Thông tin chung

- Họ và tên: Đinh Trọng Quỳnh
- Lớp: KHMT 1701
- Repo: TODO (dán link GitHub repo của bạn)
- W&B project: [csc4005-lab2-neu-cnn](https://wandb.ai/dinhtrongquynh99-dainam-vietnam/csc4005-lab2-neu-cnn)

Các run chính (để so sánh):

- Scratch: `cnn_small_scratch20` ([W&B run](https://wandb.ai/dinhtrongquynh99-dainam-vietnam/csc4005-lab2-neu-cnn/runs/ql651vq5))
- Transfer (freeze backbone): `resnet18_transfer10` ([W&B run](https://wandb.ai/dinhtrongquynh99-dainam-vietnam/csc4005-lab2-neu-cnn/runs/ipp93355))
- Fine-tune (unfreeze backbone): `resnet18_finetune10` ([W&B run](https://wandb.ai/dinhtrongquynh99-dainam-vietnam/csc4005-lab2-neu-cnn/runs/dcaeafz1))

## 2. Bài toán

Bài toán: phân loại ảnh bề mặt thép (grayscale) thành 6 loại lỗi trên bộ dữ liệu NEU Surface Defect Database:

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

Mục tiêu của lab:

- Huấn luyện 1 CNN từ đầu (from scratch)
- Thực hành transfer learning: freeze backbone và fine-tune
- So sánh dựa trên số liệu: learning curves, accuracy/loss, thời gian train/epoch, số tham số trainable

Thiết lập dữ liệu trong workspace:

- Data dir: `NEU-CLS`
- Split sizes dùng trong các run:
  - Train: 1504 ảnh
  - Val: 30 ảnh
  - Test: 266 ảnh

Lưu ý quan trọng: tập validation đang rất nhỏ (30 ảnh, 5 ảnh/lớp), nên `val_acc` có thể dao động/đạt 1.0 tương đối “dễ”. Vì vậy khi kết luận cần nhìn thêm `test_acc`, loss curves và confusion matrix.

## 3. Mô hình và cấu hình

### 3.1. MLP baseline từ Lab 1

Repo Lab 2 này không chứa kết quả MLP (Lab 1), nên phần này để TODO:

- Kiến trúc / cấu hình
- Best Val Acc / Test Acc / thời gian

### 3.2. CNN from scratch

Run: `cnn_small_scratch20`

Kiến trúc (tóm tắt):

- 3 khối ConvBlock: Conv2d(3x3) + BatchNorm + ReLU + MaxPool
- Channels: 1 → 16 → 32 → 64
- AdaptiveAvgPool2d(1x1) + MLP head (Linear 64→128→6) + Dropout(0.3)

Cấu hình train:

- Train mode: scratch
- Input: 1 channel (grayscale), normalization: none
- Image size: 64
- Optimizer: AdamW (lr=0.001, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau
- Epochs: 20, batch_size: 32, early stopping patience: 5
- Augmentation: bật (rotate nhỏ + shift + brightness/contrast)

Output artifacts (để nộp):

- `outputs/cnn_small_scratch20/best_model.pt`
- `outputs/cnn_small_scratch20/history.csv`
- `outputs/cnn_small_scratch20/curves.png`
- `outputs/cnn_small_scratch20/confusion_matrix.png`
- `outputs/cnn_small_scratch20/metrics.json`

### 3.3. Transfer learning

#### (A) Transfer – freeze backbone

Run: `resnet18_transfer10`

Thiết lập:

- Backbone: ResNet18 pretrained (ImageNet)
- Train mode: transfer (freeze backbone, chỉ train classifier head)
- Input: 3 channels (copy grayscale thành 3 kênh), normalization: ImageNet mean/std
- Image size: 128
- Optimizer: AdamW (lr=0.001, weight_decay=1e-4)
- Epochs: 10, batch_size: 32, patience: 3, augmentation: bật
- Trainable params: 3,078 (chỉ head)

Output artifacts:

- `outputs/resnet18_transfer10/best_model.pt`
- `outputs/resnet18_transfer10/curves.png`
- `outputs/resnet18_transfer10/confusion_matrix.png`
- `outputs/resnet18_transfer10/metrics.json`

#### (B) Fine-tune – unfreeze backbone

Run: `resnet18_finetune10`

Thiết lập:

- Backbone: ResNet18 pretrained
- Train mode: finetune (unfreeze backbone, train toàn bộ mạng)
- Input: 3 channels + ImageNet normalization
- Image size: 128
- Optimizer: AdamW (lr=0.0001, weight_decay=1e-4)
- Epochs: 10, batch_size: 32, patience: 3, augmentation: bật
- Trainable params: 11,179,590 (toàn bộ mạng)

Output artifacts:

- `outputs/resnet18_finetune10/best_model.pt`
- `outputs/resnet18_finetune10/curves.png`
- `outputs/resnet18_finetune10/confusion_matrix.png`
- `outputs/resnet18_finetune10/metrics.json`

## 4. Bảng kết quả

| Model | Train mode | Best Val Acc | Test Acc | Epoch time | Trainable Params | Nhận xét |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| CNN-small | scratch | 0.9000 | 0.9511 | 11.49s/epoch | 32,614 | Học ổn; nhầm Inclusion ↔ Pitted_Surface ở một số mẫu |
| ResNet18 | transfer | 1.0000 | 0.9774 | 41.72s/epoch | 3,078 | Hội tụ nhanh, accuracy cao, ít trainable params |
| ResNet18 | finetune | 1.0000 | 0.9962 | 103.39s/epoch | 11,179,590 | Accuracy cao nhất nhưng chậm nhất; cần cẩn thận overfit do val nhỏ |

Best epoch (theo val_loss):

- CNN-small scratch: epoch 17
- ResNet18 transfer: epoch 10
- ResNet18 finetune: epoch 7

## 5. Phân tích learning curves

### 5.1. CNN-small (scratch)

- Train loss giảm đều, train acc tăng dần tới ~0.96.
- Val loss có dao động (do val rất nhỏ), nhưng có xu hướng giảm và đạt tốt nhất quanh epoch 17.
- Scheduler giảm LR (0.001 → 0.0005 → 0.00025 → 0.000125) giúp ổn định về sau.

### 5.2. ResNet18 (transfer – freeze)

- Hội tụ rất nhanh: val_acc lên 1.0 từ epoch 3 và giữ ổn định.
- Val loss giảm đều tới 0.0907 ở epoch 10.
- Vì chỉ train head (3,078 params), rủi ro overfit thấp hơn và train ổn định.

### 5.3. ResNet18 (fine-tune – unfreeze)

- Train loss giảm rất nhanh và val_loss rất thấp (tốt nhất ~0.000965 ở epoch 7).
- Do val rất nhỏ, val_acc = 1.0 xuyên suốt không phản ánh hết chất lượng; tuy nhiên test_acc vẫn rất cao (0.9962).
- Thời gian/epoch cao nhất (≈ 103s/epoch) vì train toàn bộ backbone.

Các file learning curves đã được lưu trong `outputs/<run_name>/curves.png`.

## 6. Confusion matrix và lỗi dự đoán sai

### 6.1. CNN-small (scratch) – lỗi chính

Từ confusion matrix (test set 266 ảnh):

- Inclusion → Pitted_Surface: 5 ảnh
- Pitted_Surface → Inclusion: 3 ảnh
- Pitted_Surface → Crazing: 1 ảnh
- Pitted_Surface → Rolled-in_Scale: 1 ảnh
- Scratches → Inclusion: 3 ảnh

Nhận xét: Inclusion và Pitted_Surface có thể có texture tương đối giống ở một số ảnh, nên mô hình nhỏ (scratch) dễ nhầm.

### 6.2. ResNet18 (transfer)

Lỗi ít (test_acc 0.9774). Nhầm lẫn rải rác giữa:

- Crazing ↔ Patches
- Inclusion ↔ Pitted_Surface
- Rolled-in_Scale ↔ Crazing

### 6.3. ResNet18 (fine-tune)

Gần như hoàn hảo (test_acc 0.9962). Confusion matrix chỉ có 1 lỗi:

- Pitted_Surface → Inclusion: 1 ảnh

File confusion matrix:

- `outputs/cnn_small_scratch20/confusion_matrix.png`
- `outputs/resnet18_transfer10/confusion_matrix.png`
- `outputs/resnet18_finetune10/confusion_matrix.png`

## 7. Kết luận


- Transfer learning có tốt hơn không?
  - Có: ResNet18 transfer đạt test_acc 0.9774 (cao hơn scratch) và hội tụ nhanh, ổn định.
  - Fine-tune đạt cao nhất (test_acc 0.9962) nhưng tốn thời gian/epoch và trainable params lớn.

- Khi nào nên chọn transfer learning thay vì train from scratch?
  - Nên dùng transfer khi muốn baseline mạnh nhanh, hoặc dữ liệu hạn chế (NEU-CLS train ~1770 ảnh).
  - Transfer (freeze) phù hợp khi muốn tiết kiệm tài nguyên và giảm overfit (ít trainable params).
  - Fine-tune nên dùng khi cần độ chính xác tối đa và có đủ thời gian/tài nguyên; tuy nhiên cần validation đủ lớn/cross-val để tránh kết luận lệch do val quá nhỏ.
