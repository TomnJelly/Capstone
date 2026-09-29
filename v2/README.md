# v2

DB-I, DB-II **각각** 70% 학습 / 30% 검증. 1채널 마스크 U-Net, 384px. 꼭짓점은 네 변을 에지에 맞춰 서브픽셀로 뽑습니다.

v1에서 바꾼 점: 저대비 CLAHE, 더 큰 U-Net, 경계 손실, TTA, 가운데 큰 덩어리만 남김, 작은 압흔 크롭 재추론.

---

## 실행 순서

데이터(`Vicker_Database`)는 GitHub에 없습니다. 각자 받은 폴더 경로를 `--data`에 넣으세요.

```bash
git clone https://github.com/TomnJelly/Capstone.git
cd Capstone
pip install -r requirements.txt
cd v2
python preview.py --data "C:\Users\이름\Downloads\Vicker_Database"
```

미리보기 왼쪽 %는 **정답 마스크 대각선(픽셀)** 과 비교합니다. 정답·예측 둘 다 같은 에지 서브픽셀을 씁니다. 잠시 기다리면 `미리보기 사이트: http://127.0.0.1:8766/` 가 뜹니다.

사진만 올려서 예측만 보려면:

```bash
python upload.py
```

`http://127.0.0.1:8767/` 에서 파일을 고르고 예측을 누르면 됩니다. 정답 마스크는 없습니다.

DB-II:

```bash
python preview.py --split II --ckpt weights/best_II.pt --data "C:\Users\이름\Downloads\Vicker_Database"
```

가중치 `weights/best_I.pt`, `weights/best_II.pt`는 저장소에 있습니다. CPU에서는 몇 분 걸릴 수 있습니다.

---

## 데이터 폴더 모양

```
Vicker_Database/
  DB-I/
  DB-II/
  exact-masks-I/
  exact-masks-II/
```

---

## 학습 (Kaggle GPU)

1. Accelerator = **GPU T4**
2. `Vicker_Database` Add Input
3. `v2/kaggle_cell.py` 전체를 **새 셀**에 붙여 넣고 실행
4. 로그에 `device = cuda Tesla T4`
5. 끝나면 `best_I.pt`, `best_II.pt`

로컬:

```bash
python train.py
```

---

## 파일

| 파일 | 역할 |
|---|---|
| `kaggle_cell.py` | Kaggle에 붙여 넣는 학습 코드 |
| `common.py` | 데이터, U-Net, 마스크→꼭짓점 |
| `train.py` | 로컬 학습 |
| `preview.py` | 미리보기 사이트 (기본: 검증 30%) |
| `upload.py` | 사진 업로드 → 예측 마스크·꼭짓점 |
| `weights/best_I.pt` | DB-I 모델 |
| `weights/best_II.pt` | DB-II 모델 |
| `splits/` | seed 42, 70/30 목록 |
