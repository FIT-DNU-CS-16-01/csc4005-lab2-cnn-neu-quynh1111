[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/0XzoEsXD)
# CSC4005 Lab 2 – CNN Image Classification (From Scratch vs Transfer)

Starter kit này dành cho **Lab 2 của Bài 3 – CNN** trong CSC4005. Repo nối tiếp trực tiếp từ Lab 1:
- **Lab 1:** MLP + training/regularization + W&B
- **Lab 2:** CNN from scratch + transfer learning + W&B

Bài toán vẫn giữ nguyên: **phân loại 6 loại lỗi bề mặt thép** trên bộ **NEU-CLS.zip**.

## 1. Mục tiêu
Sinh viên cần:
1. train được một **CNN from scratch**,
2. chạy được một mô hình **transfer learning**,
3. dùng **W&B** để log thí nghiệm,
4. so sánh kết quả của **MLP baseline (Lab 1)**, **CNN scratch**, và **transfer learning**.

## 2. Cấu trúc repo
```text
csc4005_lab2_neu_cnn_starter/
├── README.md
├── REPORT_TEMPLATE.md
├── requirements.txt
├── configs/
│   ├── baseline_scratch.json
│   └── baseline_transfer.json
├── docs/
│   ├── GITHUB_CLASSROOM_GUIDE.md
│   ├── WANDB_GUIDE.md
│   └── LAB_GUIDE_LAB2.md
├── notebooks/
│   └── lab2_demo.ipynb
├── outputs/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
└── ci/
    ├── check_structure.py
    └── smoke_train.py
```

## 3. Cài đặt
### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows
```bash
python -m venv .venv
.venv\Scriptsctivate
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Dữ liệu
Repo này **không chứa thư mục `data/`**. Khi chạy, sinh viên phải truyền dữ liệu ngoài repo qua `--data_dir`.

Hỗ trợ các kiểu dữ liệu sau:
1. file ZIP như `NEU-CLS.zip`
2. thư mục đã giải nén từ ZIP
3. thư mục phẳng có tên file kiểu `crazing_10.jpg`, `inclusion_21.jpg`, ...
4. thư mục theo lớp `Crazing/...`, `Inclusion/...`

Ví dụ:
```bash
python -m src.train --data_dir /duong_dan/NEU-CLS.zip --model_name cnn_small --train_mode scratch
```

## 5. Chạy baseline CNN from scratch
```bash
python -m src.train   --data_dir /duong_dan/NEU-CLS.zip   --project csc4005-lab2-neu-cnn   --run_name cnn_small_baseline   --model_name cnn_small   --train_mode scratch   --optimizer adamw   --lr 0.001   --weight_decay 0.0001   --dropout 0.3   --epochs 20   --batch_size 32   --img_size 64   --patience 5   --augment   --use_wandb
```

## 6. Chạy transfer learning
Ví dụ với ResNet18:
```bash
python -m src.train   --data_dir /duong_dan/NEU-CLS.zip   --project csc4005-lab2-neu-cnn   --run_name resnet18_transfer   --model_name resnet18   --train_mode transfer   --optimizer adamw   --lr 0.001   --weight_decay 0.0001   --dropout 0.3   --epochs 10   --batch_size 32   --img_size 128   --patience 3   --augment   --use_wandb
```

Muốn fine-tune cả backbone:
```bash
python -m src.train   --data_dir /duong_dan/NEU-CLS.zip   --project csc4005-lab2-neu-cnn   --run_name resnet18_finetune   --model_name resnet18   --train_mode finetune   --optimizer adamw   --lr 0.0001   --weight_decay 0.0001   --dropout 0.3   --epochs 10   --batch_size 16   --img_size 128   --patience 3   --augment   --use_wandb
```

## 7. Output sau khi train
Mỗi run sẽ tạo thư mục:
```text
outputs/<run_name>/
```
bao gồm:
- `best_model.pt`
- `history.csv`
- `curves.png`
- `confusion_matrix.png`
- `metrics.json`

## 8. W&B
Tên project thống nhất:
```text
csc4005-lab2-neu-cnn
```

Log tối thiểu mỗi epoch:
- train_loss
- val_loss
- train_acc
- val_acc
- lr
- epoch_time_sec

Log cuối run:
- best_val_acc
- test_acc
- avg_epoch_time_sec
- total_params
- trainable_params
- confusion matrix image
- curves image

## 9. Lưu ý về transfer learning
Transfer learning trong starter kit dùng `torchvision`. Nếu gặp lỗi import `torchvision`, nguyên nhân thường là **torch / torchvision không tương thích phiên bản**.

Khi đó:
1. kiểm tra lại phiên bản torch và torchvision,
2. cài lại theo đúng cặp phiên bản,
3. vẫn có thể hoàn thành phần scratch CNN trước.

## 10. Checklist nộp bài
- [ ] Có ít nhất 1 run CNN from scratch
- [ ] Có ít nhất 1 run transfer learning
- [ ] Có W&B dashboard
- [ ] Có bảng so sánh MLP vs CNN scratch vs transfer
- [ ] Có learning curves
- [ ] Có confusion matrix
- [ ] Có kết luận khi nào transfer learning tốt hơn
