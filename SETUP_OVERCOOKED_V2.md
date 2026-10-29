# OvercookedV2 연구 환경 설정

이 fork는 upstream 환경을 모두 유지하고 OvercookedV2 실험을 추가한다.
기본 설치는 CPU용이다. 모델 가중치나 대규모 장면 데이터는 필요 없다.

## 설치

```bash
git clone https://github.com/wshf253/JaxMARL.git ~/work/overcooked-v2
cd ~/work/overcooked-v2
git remote add upstream https://github.com/bold-lab-ai/JaxMARL.git
uv sync --locked
```

이미 clone했다면 해당 폴더에서 변경 사항을 확인하고 `git pull --ff-only` 후
`uv sync --locked`를 실행한다. `upstream` 등록은 한 번만 하면 된다.
Python 버전은 `.python-version`, 의존성은 `uv.lock`으로 고정한다.

## 실행 확인

iiscluster의 로그인 노드 대신 CPU job에서 실행한다. 아래 명령은 연구실의
`lab-run` 명령이 설치된 서버용이다.

```bash
lab-run --new --gpus 0 -- env JAX_PLATFORMS=cpu uv run --locked python scripts/smoke_overcooked_v2.py
lab-run --new --gpus 0 -- env JAX_PLATFORMS=cpu uv run --locked python scripts/smoke_overcooked_v2.py --view-size 2
```

일반 PC에서는 `lab-run --new --gpus 0 --` 부분을 제외한다.
첫 실행은 JAX 컴파일로 시간이 더 걸릴 수 있다. JSON의 `status: passed`는
reset, 두 agent의 랜덤 행동, step, 종료가 동작한다는 뜻이다.
보상 0은 정상일 수 있으며, 이 검사는 학습된 정책의 성능 평가가 아니다.

옵션: `--layout cramped_room`, `--seed 0`, `--steps 100`, `--view-size 2`.
`--view-size` 생략 시 전체 격자를 관측한다. V2 자체가 항상 부분 관측을
의미하지 않으며, 작은 맵에서는 제한된 시야로도 많은 영역이 보일 수 있다.
부분 관측 실험의 planner 입력은 agent의 `obs`와 허용된 이력으로 구성한다.
전체 `state`는 평가/렌더링용으로 구분하여 정보 누출을 방지한다.

## 개발 및 결과 관리

- 환경: `jaxmarl/environments/overcooked_v2/`
- 실행 스크립트: `scripts/`
- 향후 agent/메모리 로직: `partner_adaptation/`
- 향후 실험 설정: `configs/partner_adaptation/`
- 결과: `outputs/` 또는 `results/` (Git 제외)
- 실행 중 갱신된 개인화 메모리: `runtime_memory/` (Git 제외)
- 초기 메모리 템플릿과 프롬프트는 코드와 함께 버전 관리한다.

```bash
git switch -c research/partner-adaptation
# 필요한 코드 수정 후 파일을 명시하여 git add
git diff --cached
git commit -m "Add partner adaptation experiment"
git push -u origin research/partner-adaptation
```

학습 baseline이 필요해질 때만 `uv sync --locked --extra algs`를 사용한다.
기본 smoke test에는 GPU가 필요 없다. GPU JAX는 서버 드라이버와 호환되는
CUDA extra를 별도로 선택하고 lockfile을 갱신해야 한다. 로컬 LLM은 별도
추론 환경으로 서비스하고 이 환경에서 호출하는 구성을 권장한다.

ICRL4AHT의 결과를 재현할 때는 그 프로젝트의 환경 commit과 설정을 별도로
맞춰야 한다. 이 fork의 기본 실행은 ICRL4AHT 재현 실험이 아니다.
