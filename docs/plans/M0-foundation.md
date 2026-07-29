# M0 — 專案地基

**目標**：建立可運作的套件骨架與公開 API 型別定義，讓 M1 能直接開始寫 Perceive-Verify Loop。
**前置**：無。
**估時**：1 週
**參考**：`docs/specs/2026-07-29-saccade-design.md`

---

## 驗收條件（全部通過才進 M1）

```bash
uv run pytest                    # 全綠
uv run ruff check src tests      # 無錯誤
uv run ruff format --check .     # 格式一致
uv run mypy src                  # 無錯誤
uv build                         # 套件可建置
```

外加：
- `python -c "import saccade; print(saccade.__all__)"` 列出所有公開名稱
- GitHub Actions CI 綠燈
- `pre-commit run --all-files` 通過

---

## Task 1：uv 與 Git 初始化

**做什麼**

1. 安裝 uv（若尚未安裝）：
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. 初始化專案：
   ```bash
   uv init --lib --name saccade-vision
   ```
   注意：`--lib` 產生 library 佈局（src layout），非 application。

3. `git init`（預設分支 `main`）

4. `.gitignore`：
   ```
   .env
   __pycache__/
   *.pyc
   .venv/
   .pytest_cache/
   .mypy_cache/
   .ruff_cache/
   node_modules/
   dist/
   build/
   *.egg-info/
   cache/
   data/videos/
   *.mp4
   *.mov
   models/*.pt
   ```
   注意：**`uv.lock` 必須進 git**，不可 ignore。

5. `.python-version`：`3.11`

6. `.env.example`：
   ```
   # Pydantic AI 讀取的變數名稱，勿改
   GOOGLE_API_KEY=
   OPENAI_API_KEY=
   SACCADE_CACHE_DIR=./cache
   ```
   注意：Gemini 對應的環境變數是 `GOOGLE_API_KEY`，不是 `GEMINI_API_KEY`。

7. 首個 commit：`docs/` + 設定檔

**驗證**：`git log` 一筆 commit；`uv --version` 有輸出。

---

## Task 2：pyproject.toml

**做什麼**

依 spec 4.5 節設定：

```toml
[project]
name = "saccade-vision"
version = "0.1.0"
description = "Active vision agent — give VLMs a saccade system"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2",
    "pydantic-ai>=1.0",
    "pillow>=10",
    "numpy>=1.26",
]

[project.optional-dependencies]
geometry = ["opencv-python>=4.10"]
observability = ["langfuse>=2"]

[dependency-groups]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.24",
    "pytest-cov",
    "ruff",
    "mypy",
    "pre-commit",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM", "RUF"]

[tool.mypy]
strict = true
python_version = "3.11"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

同步環境：
```bash
uv sync --all-extras --dev
```

**驗證**：`uv run python -c "import saccade"` 不報錯；`uv.lock` 已產生。

---

## Task 3：公開資料模型

**檔案**：`src/saccade/models.py`

Pydantic v2 定義（spec 4.2 列出的公開型別）：

| 型別 | 欄位 |
|---|---|
| `BBox` | `x, y, w, h: int` |
| `Viewport` | `bbox: BBox`、`zoom: float`、`source_size: tuple[int, int]` |
| `Observation` | `statement: str`、`self_confidence: float \| None` |
| `Verification` | `passed: bool`、`method: str`、`computed: dict[str, Any]`、`conflict: str \| None` |
| `EvidenceStep` | `index: int`、`action_name: str`、`viewport: Viewport`、`observation: Observation`、`verification: Verification \| None`、`image_ref: str \| None` |
| `InvestigationResult` | `answer: str`、`confidence: float`、`converged: bool`、`evidence_chain: list[EvidenceStep]`、`total_tokens: int`、`structured: Any \| None` |
| `VLMResponse` | `text: str`、`confidence: float \| None`、`raw: dict`、`tokens_used: int`、`model_id: str`、`structured: Any \| None` |

**注意**：
- `InvestigationResult` 需支援泛型（`expect=` 傳入的型別）。用 `Generic[T]` 或 `structured: Any`，M0 先用後者，M1 再視需要泛型化。
- 此檔案只 import `pydantic`、`typing`。不得 import PIL、cv2、numpy。

**驗證**：`tests/test_models.py` — 每型別建構一次、`model_dump_json()` 一次、非法值拋 `ValidationError`。

---

## Task 4：Port 與 Tool 定義

**檔案**：`src/saccade/ports.py`、`src/saccade/tools.py`

`ports.py` — 用 `typing.Protocol` + `@runtime_checkable`：
```python
@runtime_checkable
class VLMPort(Protocol):
    @property
    def model_id(self) -> str: ...
    async def ask(
        self, images: list[Image.Image], prompt: str, output_type: type | None = None
    ) -> VLMResponse: ...

@runtime_checkable
class CachePort(Protocol):
    def get(self, key: str) -> VLMResponse | None: ...
    def set(self, key: str, value: VLMResponse) -> None: ...
```

**注意（spec 4.3）**：Port 簽章收 `PIL.Image.Image`，但 Pydantic AI 實際接受的是
`BinaryContent(data=bytes, media_type=str)`。轉換由 `vlm/pydantic_ai.py` adapter 負責，
是唯一知道 `BinaryContent` 存在的地方。M0 只定簽章，M1 實作 adapter 時處理轉換。

`tools.py`：
```python
@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    fn: Callable[..., ToolResult]
    params_schema: type[BaseModel]

@dataclass
class ToolResult:
    value: Any
    is_measurement: bool          # 見 spec 3.3：決定能否對質 VLM
    evidence_image: Image.Image | None = None
```

**驗證**：`tests/test_ports.py` — 寫假實作，`isinstance(fake, VLMPort)` 為 True。

---

## Task 5：例外階層

**檔案**：`src/saccade/exceptions.py`

依 spec 4.4：`SaccadeError` → `VLMError`、`ToolError`。

**驗證**：`tests/test_exceptions.py` — 確認繼承關係，`except SaccadeError` 能攔到子類。

---

## Task 6：`__init__.py` 與公開 API

**檔案**：`src/saccade/__init__.py`、`src/saccade/py.typed`

依 spec 4.2 寫 import 與 `__all__`。`agent.py` 此時只需空殼：

```python
# src/saccade/agent.py — M0 只定簽章，M1 實作
class ActiveVisionAgent:
    def __init__(self, vlm, *, cache=None, max_steps=8,
                 confidence_threshold=0.8, tools=None, on_step=None) -> None:
        raise NotImplementedError("M1")
```

`py.typed` 為**空檔案**，但必須存在（PEP 561），否則使用者的 mypy 看不到型別。
確認 `pyproject.toml` 有把它打包進去（hatchling 預設會含 `src/` 下所有檔案）。

**驗證**：
```bash
uv run python -c "import saccade; assert len(saccade.__all__) > 10; print(saccade.__version__)"
```

---

## Task 7：FakeVLM 與檔案快取

**檔案**：`src/saccade/vlm/fake.py`、`src/saccade/vlm/_cache.py`

**FakeVLM**（公開，使用者也能用來測自己的整合）：
```python
class FakeVLM:
    def __init__(self, responses: list[str] | list[BaseModel]): ...
    async def ask(self, images, prompt, output_type=None) -> VLMResponse: ...
    @property
    def calls(self) -> list[tuple]: ...     # 供測試斷言
```

**FileCache**：
- 鍵 = `sha256(圖片 bytes 串接 + prompt + model_id)`
- 存 JSON 於 `SACCADE_CACHE_DIR`
- 目錄不存在時自動建立

**驗證**：
- `tests/test_fake_vlm.py` — 回應依序、calls 記錄正確、超出清單長度時的行為明確
- `tests/test_cache.py` — set/get 往返、不同圖產生不同鍵、用 `tmp_path` fixture 隔離

---

## Task 8：幾何計算

**檔案**：`src/saccade/geometry/shapes.py`

M1 的 Verifier 會用到。先實作 BlindTest 相關基礎：

| 函式 | 用途 |
|---|---|
| `circles_overlap(c1, r1, c2, r2) -> bool` | 圓心距 vs 半徑和 |
| `count_line_intersections(lines) -> int` | 線段交點計數 |
| `distance(p1, p2) -> float` | 歐氏距離 |

**opencv 選配處理**：`geometry/__init__.py` 需在 import 失敗時給清楚訊息：
```python
try:
    import cv2
except ImportError as e:
    raise ImportError(
        '幾何驗證需要 opencv。請執行：pip install "saccade-vision[geometry]"'
    ) from e
```

**硬規則**：此模組**不得**呼叫 `cv2.imread`、`cv2.VideoCapture`、`Image.open`、`Image.save`。只接收記憶體物件。

**驗證**：`tests/test_geometry.py` — 每函式至少 3 案例（正常／邊界／退化）。重疊判定必須測「剛好相切」。另測未安裝 opencv 時的錯誤訊息。

---

## Task 9：架構守衛測試

**檔案**：`tests/test_architecture.py`

**這是防技術債的自動化機制**，對應 spec 第 11 節規則 2、3。

用 AST 掃描 `src/saccade/`：

**檢查 1：禁止 import 領域套件**
```python
FORBIDDEN_IMPORTS = {"mediapipe", "ultralytics", "sklearn", "torch", "tensorflow"}
```

**檢查 2：純邏輯模組禁止 I/O 呼叫**
```python
PURE_MODULES = ["_planner.py", "_verifier.py", "_evidence.py", "geometry/"]
FORBIDDEN_CALLS = {
    "cv2.imread", "cv2.imwrite", "cv2.VideoCapture",
    "Image.open", "Image.save", "open",
}
```

**檢查 3：`__all__` 內的名稱都必須存在且可 import**

**驗證**：故意在 `geometry/shapes.py` 加 `cv2.imread(...)` → 測試失敗；移除後通過。

---

## Task 10：pre-commit

**檔案**：`.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: local
    hooks:
      - id: mypy
        name: mypy
        entry: uv run mypy src
        language: system
        pass_filenames: false
      - id: architecture
        name: architecture guard
        entry: uv run pytest tests/test_architecture.py -q
        language: system
        pass_filenames: false
```

安裝：`uv run pre-commit install`

**為什麼**：架構守衛若只在 CI 跑，會 commit 完才發現違規。pre-commit 當場擋。

**驗證**：`uv run pre-commit run --all-files` 通過。

---

## Task 11：CI

**檔案**：`.github/workflows/ci.yml`

觸發：push 到 `main`、所有 PR。

```yaml
jobs:
  test:
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --all-extras --dev
      - run: uv run ruff check src tests
      - run: uv run ruff format --check .
      - run: uv run mypy src
      - run: uv run pytest --cov=saccade --cov-report=term-missing
      - run: uv build
```

**驗證**：推上 GitHub 後 Actions 綠燈。

---

## Task 12：README

**檔案**：`README.md`

### 寫作要求（重要）

**要有人味，不要 AI 樣板。** 具體避免：
- `🚀 Features` / `✨ Highlights` 這類 emoji 條列
- 「本專案旨在…」「致力於…」公文腔
- 貼一整棵目錄樹（沒人看）
- 「Contributions are welcome!」罐頭句
- 預告還沒做出來的成果

**要有的**：
- 開頭直接講「VLM 為什麼是瞎的」這個具體故事，附數字與論文連結
- 為什麼做這個（讀到 position paper 說沒人做，決定用推論階段的方式試試）
- 誠實標記現在做到哪、什麼還不行
- 一段最小可用範例（M1 後補）

**所有數字必須有出處**。M0 階段可寫的已查證數字：

| 數字 | 出處 |
|---|---|
| 58.07% / 77.84% / 56.84% | VLMs Are Blind 論文 |
| 「近視眼／眼盲」比喻 | 同上，論文原文 |
| position paper 指出 unexplored | 論文原文 |

**BlindTest 的提升數字 M2 才寫，M0 不得預告。**

### M0 版結構

```markdown
# Saccade

（一段：VLM 在小學生等級的視覺任務上只有 58%，
  問題不在看不到，在不會自己決定要看哪裡。引用論文。）

（一段：人眼靠 saccade 跳視取樣，VLM 是看一眼就答。
  這個套件在推論階段補上那套跳視，不用訓練。）

## 現在的狀態
M0 完成 — 骨架與型別定義。核心 loop 開發中。

## 架構
（Perceive-Verify Loop 的圖，五個步驟）
（一句話說明：工具在這裡是 VLM 的裁判，不是延伸）

## 安裝
uv add saccade-vision
# 需要幾何驗證：
uv add "saccade-vision[geometry]"

## 開發
uv sync --all-extras --dev
uv run pytest

## Roadmap
- [x] M0 骨架
- [ ] M1 核心 loop + 第一個 BlindTest 數字
- [ ] M2 完整 benchmark
- ...
```

---

## 完成後

1. 跑一次全部驗收條件，**貼出實際輸出**（不要只說「通過」）
2. commit + push，確認 CI 綠
3. **產出 M1 plan**，再繼續

---

## M0 不做的事

- 任何 loop 邏輯（M1）
- 真實 VLM adapter（M1）
- `apps/sandlot-baseball/`（M3）
- MediaPipe / YOLO（M3）
- FastAPI / 前端（M6）
- 資料庫（M3）
