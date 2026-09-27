# 비커스 압흔 측정 AI

현미경으로 찍은 비커스(◆) 압흔 사진에서 **마스크**를 찾고, 마스크 끝점으로 **네 꼭짓점**을 뽑습니다.

1채널 3단 U-Net, 256px. 꼭짓점은 마스크의 N/E/S/W입니다.

---

## 다른 컴퓨터에서 미리보기

데이터(`Vicker_Database`)는 GitHub에 없습니다. 각자 받은 폴더 경로를 `--data`에 넣으세요.

```bash
git clone https://github.com/TomnJelly/Capstone.git
cd Capstone
pip install -r requirements.txt
python preview.py --data "C:\Users\이름\Downloads\Vicker_Database"
```

잠시 기다리면 터미널에 `미리보기 사이트: http://127.0.0.1:8766/` 가 뜹니다. 그다음에 브라우저가 열립니다. 안 열리면 그 주소를 주소창에 직접 넣으세요. 끄려면 터미널에서 `Ctrl+C`.

가중치 `weights/best.pt`는 저장소에 있습니다. 50장을 만들므로 CPU에서는 몇 분 걸릴 수 있습니다. 그 동안은 사이트가 아직 없습니다.

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
3. `kaggle_cell.py` 전체를 **새 셀**에 붙여 넣고 실행
4. 로그에 `device = cuda Tesla T4`
5. 끝나면 `best.pt`와 셀 아래 검증 8장

---

## 파일

| 파일 | 역할 |
|---|---|
| `kaggle_cell.py` | Kaggle에 붙여 넣는 학습 코드 |
| `common.py` | 데이터, U-Net, 마스크→꼭짓점 |
| `train.py` | 로컬 학습 |
| `preview.py` | 미리보기 사이트 |
| `weights/best.pt` | 팀 모델 |
