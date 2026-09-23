# 비커스 압흔 측정 AI

현미경으로 찍은 비커스(◆) 압흔 사진에서 **압흔 영역**과 **네 꼭짓점**을 찾고, 대각선 길이로 경도를 계산하기 위한 학습 코드입니다.

Flutter 앱(`flutter_application_3`)과는 별도입니다. 이 저장소는 **학습·검증**만 담당하고, 완성된 가중치(`.pt`)는 넣지 않습니다.

---

## 팀원이 할 일 (요약)

1. 이 저장소는 **비공개**입니다. 초대받은 뒤 클론하세요.
2. Jalilian `Vicker_Database`를 각자 PC(또는 Kaggle Input)에 둡니다. 데이터는 용량·배포 조건 때문에 GitHub에 없습니다.
3. **학습은 Kaggle GPU**에서 합니다. 로컬은 미리보기용입니다.
4. Kaggle이 만든 `best.pt`를 `weights/best.pt`에 넣고, `preview.py`로 결과를 확인합니다.

```bash
git clone https://github.com/TomnJelly/Capstone.git
cd Capstone
pip install -r requirements.txt
```

---

## 데이터

[요청 페이지](https://www.wavelab.at/sources/Jalilian24a/)에서 받은 `Vicker_Database` 폴더가 아래처럼 보이면 됩니다.

```
Vicker_Database/
  DB-I/                          학습용 사진
  DB-II/                         검증용 사진
  exact-masks-I/                 학습용 정답 마스크
  exact-masks-II/                검증용 정답 마스크
  04_Algorithmus_Statistik/ground_truth/
    db1_measure_lines_serie1.txt 학습용 정답 꼭짓점
    db2_measure_lines_serie1.txt 검증용 정답 꼭짓점
```

쓰는 정답은 두 가지입니다.

| 무엇을 | 어디서 | 왜 |
|---|---|---|
| 압흔 모양 | `exact-masks-*` | 사람이 칠한 영역 |
| 네 꼭짓점 N/E/S/W | `db*_measure_lines_serie1.txt` | Jalilian 평가용 측정선. 네 변의 중점 |

`Manual_part` XML의 `NPX`/`NPY`는 사진 좌표와 어긋난 장이 있어서 **쓰지 않습니다.**

---

## 학습 (Kaggle, 권장)

GPU가 필요합니다. 노트북 한 칸에 `kaggle_cell.py` **파일 전체**를 붙여 넣으면 됩니다.

1. 새 Kaggle Notebook을 만들고 GPU를 켭니다.
2. `Vicker_Database`를 **Add Input** 합니다. 위에 적은 폴더가 Input 안에 있어야 합니다.
3. `kaggle_cell.py`를 셀 하나에 그대로 붙여 넣고 실행합니다.
4. 끝나면 `/kaggle/working/best.pt`를 받아, 이 저장소의 `weights/best.pt`로 복사합니다.

설정은 코드와 같아야 합니다.

- 입력 크기 **384**
- 출력 **5채널** (`unet4_heat5`: 마스크 1 + 꼭짓점 히트맵 4)
- **40 epoch**
- 저장 기준: 검증 Dice가 0.85 이상일 때, **대각선 오차**가 더 작은 체크포인트

`best.pt`가 이 구조가 아니면 `preview.py`가 열지 않습니다. 예전에 학습한 1채널/256 모델은 쓰지 마세요.

---

## 미리보기 (로컬)

학습이 잘 됐는지 50장을 HTML로 봅니다. `--data`는 본인 PC의 `Vicker_Database` 경로로 바꾸세요.

```bash
python preview.py --data "C:\path\to\Vicker_Database" --ckpt weights/best.pt
python -m http.server 8766 --directory preview
```

브라우저에서 `http://localhost:8766` 을 엽니다.

- 왼쪽: 장마다 대각선 오차
- 위: 전체 평균/중앙값, 좋음·보통·나쁨 개수
- 사진 위 글씨는 작게 두는 것이 맞습니다. 오차는 옆 목록에서 봅니다.

로컬에서 직접 학습할 때만:

```bash
python train.py --data "C:\path\to\Vicker_Database"
```

CPU에서는 매우 느립니다. 팀 모델은 Kaggle에서 나온 `best.pt`를 씁니다.

---

## 파일

| 파일 | 역할 |
|---|---|
| `kaggle_cell.py` | Kaggle에 붙여 넣는 학습 코드 (이 파일만 있으면 됨) |
| `common.py` | 데이터 읽기, U-Net, 손실, 추론 |
| `train.py` | 로컬에서 돌릴 때. `common.py`를 씀 |
| `preview.py` | 결과 HTML |
| `requirements.txt` | Python 패키지 |
| `weights/` | `best.pt`를 두는 곳. Git에는 올리지 않음 |

`train.py`와 `kaggle_cell.py`의 모델·정답·설정은 같아야 합니다. 학습 코드를 고치면 **둘 다** 맞추고, Kaggle 셀도 다시 붙여 넣으세요.

---

## 모델이 하는 일

1. 사진을 384×384로 넣습니다.
2. U-Net이 마스크와 꼭짓점 4개 위치를 같이 예측합니다.
3. 압흔 근처를 한 번 더 잘라서 꼭짓점을 다듬습니다.
4. d1(N–S), d2(E–W)는 **원본 사진 픽셀**입니다. 크롭은 보기 창일 뿐입니다.

USB 현미경처럼 보이게 만드는 함수(`simulate_uvc_*`)는 예시용입니다. **학습 증강에는 넣지 않았습니다.**
