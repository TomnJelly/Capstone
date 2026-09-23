# 비커스 압흔 측정 AI

현미경으로 찍은 비커스(◆) 압흔 사진에서 **마스크**를 찾고, 마스크 끝점으로 **네 꼭짓점**을 뽑습니다.

 1채널 3단 U-Net, 256px, 꼭짓점은 마스크 N/E/S/W.

---

## 로직

1. 초대받은 뒤 클론하세요.
2. Jalilian `Vicker_Database`를 각자 PC(또는 Kaggle Input)에 둡니다.
3. 학습은 Kaggle **GPU (T4)** 에서 합니다.
4. `/kaggle/working/best.pt`를 `weights/best.pt`에 넣고 `preview.py`로 확인합니다.

```bash
git clone https://github.com/TomnJelly/Capstone.git
cd Capstone
pip install -r requirements.txt
```

---

## 데이터

```
Vicker_Database/
  DB-I/              학습용 사진
  DB-II/             검증용 사진
  exact-masks-I/     학습용 정답 마스크
  exact-masks-II/    검증용 정답 마스크
```

꼭짓점 정답 파일은 쓰지 않습니다. 사람 마스크의 맨 위/오른쪽/아래/왼쪽이 N/E/S/W입니다.

---

## 학습 (Kaggle GPU)

1. Accelerator = **GPU T4**
2. `Vicker_Database` Add Input
3. `kaggle_cell.py` 전체를 새 셀에 붙여 넣고 실행
4. 로그에 `device = cuda Tesla T4` 가 나와야 합니다
5. 끝나면 `best.pt`와 셀 아래 검증 8장(왼쪽 정답, 오른쪽 예측)

30 epoch, 256px, 1채널. `val_dice`가 제일 좋은 체크포인트를 저장합니다.

---

## 미리보기 (로컬)

```bash
python preview.py --data "C:\path\to\Vicker_Database" --ckpt weights/best.pt
python -m http.server 8766 --directory preview
```

---

## 파일

| 파일 | 역할 |
|---|---|
| `kaggle_cell.py` | Kaggle에 붙여 넣는 학습 코드 |
| `common.py` | 데이터, U-Net, 마스크→꼭짓점 |
| `train.py` | 로컬 학습 |
| `preview.py` | 결과 HTML |
| `weights/best.pt` | 팀 모델 |
