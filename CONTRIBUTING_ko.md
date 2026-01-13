# Contributing to NoiseChain

NoiseChain 프로젝트에 기여해 주셔서 감사합니다!

## 개발 환경 설정

```bash
# 저장소 클론
git clone https://github.com/sadpig70/NoiseChain.git
cd NoiseChain

# 가상 환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 개발 의존성 설치
pip install -e ".[dev]"

# 테스트 실행
pytest
```

## 코드 스타일

- **Formatter**: Ruff
- **Type Checker**: mypy
- **Line Length**: 100자

```bash
# 코드 검사
ruff check src/

# 타입 검사
mypy src/noisechain/
```

## 커밋 메시지 규칙

```
<type>: <subject>

<body>
```

### Type

- `feat`: 새 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `test`: 테스트 추가/수정
- `refactor`: 리팩토링
- `chore`: 빌드, 설정 변경

### 예시

```
feat: Add correlation signature verification

- Implement CorrelationSignature.verify()
- Add cosine similarity threshold
- Update tests
```

## Pull Request 절차

1. Fork 후 feature 브랜치 생성
2. 변경 사항 커밋
3. 테스트 통과 확인 (`pytest`)
4. PR 생성

## 문의

- 📧 <sadpig70@gmail.com>
- 🔗 [GitHub Issues](https://github.com/sadpig70/NoiseChain/issues)
