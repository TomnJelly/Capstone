# 비커스 압흔 측정 AI

- `v1`: 처음 잘 나온 1채널 버전
- `v2`: DB-I / DB-II 각각 70% 학습, 30% 검증으로 더 학습한 버전

데이터(`Vicker_Database`)는 GitHub에 없습니다.

```bash
git clone https://github.com/TomnJelly/Capstone.git
cd Capstone
pip install -r requirements.txt
cd v2
python preview.py --data "C:\Users\이름\Downloads\Vicker_Database"
```

잠시 기다리면 `미리보기 사이트: http://127.0.0.1:8766/` 가 뜹니다. 자세한 실행은 `v2/README.md`, 이전 버전은 `v1/README.md`.
