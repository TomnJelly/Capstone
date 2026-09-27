# v2

DB-I, DB-II **각각** 70% 학습 / 30% 검증. 1채널 마스크 U-Net, 384px. 꼭짓점은 마스크의 N/E/S/W입니다.

v1에서 바꾼 점: 저대비 CLAHE, 더 큰 U-Net, 경계 손실, TTA, 가운데 큰 덩어리만 남김.

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

기본은 **검증 30%**만 보여줍니다. 잠시 기다리면 터미널에 `미리보기 사이트: http://127.0.0.1:8766/` 가 뜹니다. 안 열리면 그 주소를 주소창에 직접 넣으세요. 끄려면 `Ctrl+C`.

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
| `weights/best_I.pt` | DB-I 모델 |
| `weights/best_II.pt` | DB-II 모델 |
| `splits/` | seed 42, 70/30 목록 |
