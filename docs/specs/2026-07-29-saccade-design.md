# Saccade — 主動視覺 Agent 設計文件

**日期**：2026-07-29
**狀態**：已核准，待實作
**作者**：yslin226

---

## 1. 專案動機

### 1.1 問題

視覺語言模型（VLM）在低階視覺任務上的表現遠低於人類。

論文《Vision Language Models Are Blind》（Rahmanzadehgervi et al.）以 7 個小學生等級的任務（兩圓是否重疊、兩線交叉幾次、單字中哪個字母被圈起、奧運標誌有幾個圈）測試四個頂尖 VLM：

| 指標 | 數值 |
|---|---|
| 四個頂尖 VLM 平均正確率 | **58.07%** |
| 最佳（Claude 3.5 Sonnet） | 77.84% |
| 線條交叉計數 | 56.84%（近似隨機） |
| 人類 | 100% |

論文結論：「它們的視覺，好一點像近視眼看細節模糊，差一點像一個聰明但眼盲的人在合理猜測。」

### 1.2 根因

論文《Position: The Systemic Lack of Agency in Visual Reasoning》（2026）指出根本原因：**VLM 是被動的**。

現行 VLM 的缺失：
- 無法主動控制視覺注意力（不能決定「下一步該看哪」）
- 無法迭代探索場景（只能單次前向傳遞，不能來回看）
- 缺乏將推理需求轉為視覺行動的能力

論文呼籲的方向：主動視覺系統、迭代視覺探索框架、能動性感知架構。論文自述這些方向 **unexplored**。

補充佐證：《QuantiPhy》（2026）指出 VLM「尚未在視覺觀察與量化物理事實之間建立可靠連結，比較像是根據語意脈絡做近似猜測的猜測者，而不是精確的視覺測量者」。

### 1.3 本專案的切入角度

上述論文提出的解法多為**改架構、用 RL 訓練**（需大量算力）。

**Saccade 走另一條路：在推論階段（inference-time）以 agent 補足能動性，零訓練成本。**

命名由來：人類視覺並非「看一眼就懂」。眼睛每秒進行 3–4 次跳視（saccade），快速跳到不同位置取樣，由大腦整合。VLM 目前正是「看一眼就回答」，這是它失明的原因。Saccade 為 VLM 補上這套跳視能力。

### 1.4 核心設計原則

> **系統不被允許說出任何沒有數字支撐的話。**

技術落實：**VLM 被關在「解釋」的角色，測量由可計算的工具負責。**

```
❌ 禁止：影片 → VLM → 「你重心偏前」
✅ 要求：影片 → 幾何計算 → 「重心於觸擊前 0.12s 越過前腳踝 3.2cm」
                → VLM 驗證數字合理性 + 翻譯成人話 + 附證據幀
```

此原則是本專案與「把影片丟給通用 AI 問意見」的根本差異。後者輸出無數字、不可驗證、不可重現。

---

## 2. 產出物與範圍

### 2.1 兩個產出物

| 產出 | 類型 | 架構風格 | 定位 |
|---|---|---|---|
| **Saccade** | Python 函式庫（library） | Library 慣例（扁平、公開 API 明確） | 通用主動視覺 Agent |
| **Sandlot Baseball** | 應用程式（application） | Clean Architecture | 棒球動作分析工具 |

**兩者架構風格刻意不同。** Clean Architecture 是為應用程式設計的（有使用者、有請求、有資料庫、有 use case 流程）；函式庫沒有這些，其使用者是另一個程式。主流 Python 函式庫（requests、pydantic、httpx）皆採扁平結構加明確公開 API，不採應用分層。

### 2.2 引擎與應用的邊界

**Saccade 提供**：
- Perceive-Verify Loop（決定看哪→改變觀看→觀察→驗證→收斂）
- 視覺動作（裁切／放大／標註）
- 幾何驗證（以 OpenCV 交叉檢查 VLM 陳述）
- 證據鏈（每步驟的截圖、數字、理由）
- VLMPort / CachePort（可自訂實作）
- 工具註冊機制
- BlindTest 評測

**Saccade 不提供**（由應用負責）：
- 領域知識（棒球力學、醫學、PCB 規格）
- 領域偵測器（MediaPipe、YOLO）
- 業務流程、UI、儲存
- RAG 知識庫

**硬性規定**：**Saccade 不得 import MediaPipe、YOLO 或任何領域專用套件**。領域能力一律由應用透過 `register_tool()` 注入。違反此規則將使引擎失去通用性，BlindTest 亦無法共用同一核心。

### 2.3 明確不做（YAGNI）

- 手機原生 App
- 即時串流分析
- 使用者帳號／多租戶
- 模型訓練或微調
- 商業化功能

---

## 3. Agent 架構

### 3.1 定位：ReAct + Reflexion 變體

| 既有架構 | 借用什麼 | 差異 |
|---|---|---|
| **ReAct** | Reason → Act → Observe 的骨架 | ReAct 的 Act 是向外求助取得新資訊；Saccade 的 Act **同時包含「改變觀看方式」與「呼叫工具」** |
| **Reflexion** | 對輸出做批判與修正 | Reflexion 通常是 LLM 批判 LLM（同一個瞎子評自己）；**Saccade 的批判者是可計算的工具，不是 LLM** |

**核心差異一句話**：一般 agent 把工具當作 VLM 的延伸（相信工具回傳的資訊）；**Saccade 把工具當作 VLM 的裁判**（用工具結果對質 VLM 陳述）。

範例：
| 情境 | 一般 ReAct | Saccade |
|---|---|---|
| VLM 說「兩圓重疊」 | 接受 | OpenCV 算圓心距 47px < 半徑和 52px → 確認 |
| VLM 說「三個人」 | 接受 | YOLO 偵測到 2 人 → **衝突，降低信心，放大重數** |

### 3.2 Perceive-Verify Loop

```
輸入：image, question, tools

初始化：viewport = 全圖；evidence = []；confidence = 0

迴圈（直到 confidence >= 門檻 或 達 max_steps）：

  1. [Plan]    依 question、已探索區域、目前信心，決定下一步視野與工具
  2. [Act]     執行動作（見 3.3 三類工具）
  3. [Observe] VLM 觀察處理後的視野，輸出結構化陳述
  4. [Verify]  以 is_measurement=True 的工具結果對質 VLM 陳述
                 一致 → 提高信心
                 衝突 → 降低信心，記錄衝突，回到 1 換角度
  5. [Record]  記錄本步驟：截圖、工具、數字、陳述、驗證結果

輸出：InvestigationResult(answer, confidence, converged, evidence_chain)
```

**Loop 專屬狀態**（此為自行實作的部分，現有框架未提供抽象）：
- **視野管理**：目前區域、放大倍率、已探索區域集合
- **衝突仲裁**：VLM 陳述與測量結果不一致時的裁決規則
- **證據鏈**：每步驟可回溯的截圖與數字
- **信心收斂**：停止條件（門檻／步數上限／連續無新資訊）

### 3.3 三類工具

| 類型 | 內容 | is_measurement |
|---|---|---|
| **視野操作** | crop、zoom、rotate、enhance_contrast、annotate | — |
| **計算工具** | measure_geometry、count_contours、pixel_stats | ✅ True |
| **領域工具**（應用註冊） | pose_estimate、detect_objects、compare_frames、search_knowledge | 由註冊者宣告 |

`is_measurement` 決定 Verifier 能否以該結果對質 VLM。VLM 產生的描述性工具結果為 `False`（只是另一個意見，不能當裁判）。

### 3.4 框架分工

| 職責 | 由誰負責 | 理由 |
|---|---|---|
| 模型呼叫、結構化輸出保證、工具 schema 生成、重試 | **Pydantic AI** | 標準問題，重造輪子無意義。2026 起手刻 while loop 呼叫 LLM 已非正式環境做法 |
| 視野管理、衝突仲裁、證據鏈、信心收斂 | **自行實作** | 視覺專屬狀態，現有框架無對應抽象。此為本專案創新本體 |

**不使用 litellm**：Pydantic AI 原生支援 OpenAI、Anthropic、Gemini、Groq、Mistral、Cohere、Bedrock，以及**任何 OpenAI 相容端點**。GLM、Qwen、OpenRouter、Ollama、vLLM 皆提供 OpenAI 相容 API，以 `OpenAIModel(base_url=...)` 即可接入。多一層 litellm 無實益。

---

## 4. Saccade 引擎設計（Library）

### 4.1 目錄結構

```
src/saccade/
├── __init__.py           # 公開 API + __all__ + __version__
├── py.typed              # PEP 561 型別標記（空檔案，必要）
├── agent.py              # ActiveVisionAgent — 主入口
├── models.py             # 公開資料模型
├── ports.py              # VLMPort, CachePort
├── tools.py              # Tool, ToolResult
├── exceptions.py         # 例外階層
├── _planner.py           # 內部：決定下一步
├── _observer.py          # 內部：呼叫 VLM
├── _verifier.py          # 內部：衝突仲裁
├── _evidence.py          # 內部：證據鏈
├── actions/              # 視覺動作（公開，可自訂）
│   ├── __init__.py
│   ├── crop.py
│   ├── zoom.py
│   └── annotate.py
├── geometry/             # 幾何計算（公開，需 [geometry] extra）
│   ├── __init__.py
│   └── shapes.py
└── vlm/                  # 內建 VLM 實作
    ├── __init__.py
    ├── pydantic_ai.py
    └── fake.py           # 公開，供使用者測試用
```

**命名慣例**（PEP 8 標準）：底線前綴 `_module.py` 表示內部實作，可自由修改而不構成 breaking change。無底線者為公開契約。

### 4.2 公開 API

```python
# src/saccade/__init__.py
from saccade.agent import ActiveVisionAgent
from saccade.models import (
    BBox, Viewport, Observation, Verification,
    EvidenceStep, InvestigationResult, VLMResponse,
)
from saccade.ports import VLMPort, CachePort
from saccade.tools import Tool, ToolResult
from saccade.exceptions import SaccadeError, VLMError, ToolError

__version__ = "0.1.0"

__all__ = [
    "ActiveVisionAgent",
    "BBox", "Viewport", "Observation", "Verification",
    "EvidenceStep", "InvestigationResult", "VLMResponse",
    "VLMPort", "CachePort", "Tool", "ToolResult",
    "SaccadeError", "VLMError", "ToolError",
]
```

公開介面刻意維持精簡。`__all__` 同時作為人類可讀的公開清單，並讓型別檢查器辨識 re-export。

### 4.3 主入口簽章

```python
class ActiveVisionAgent:
    def __init__(
        self,
        vlm: VLMPort | str,
        *,                                     # 之後參數強制 keyword-only
        cache: CachePort | None = None,
        max_steps: int = 8,
        confidence_threshold: float = 0.8,
        tools: list[Tool] | None = None,
        on_step: Callable[[EvidenceStep], None] | None = None,
    ) -> None: ...

    async def investigate_async(
        self,
        image: Image.Image,
        question: str,
        *,
        expect: type[T] | None = None,
    ) -> InvestigationResult[T]: ...

    def investigate(
        self,
        image: Image.Image,
        question: str,
        *,
        expect: type[T] | None = None,
    ) -> InvestigationResult[T]:
        """同步版本，內部包裝 async 實作。"""

    def register_tool(self, tool: Tool) -> None: ...
```

**設計決策**：

| 決策 | 內容 | 理由 |
|---|---|---|
| **keyword-only 參數** | `*` 之後全部強制具名 | 日後新增參數不影響既有呼叫，不構成 breaking change |
| **async 為核心，sync 為包裝** | `investigate_async` 實作、`investigate` 包裝 | VLM 呼叫為網路 I/O，async 為正確模型；反向（sync 核心包 async）技術上做不到 |
| **`vlm` 接受字串或 Port** | `ActiveVisionAgent("google:gemini-2.5-flash")` 或注入自訂實作 | 降低入門門檻，同時保留完全控制 |
| **`on_step` callback** | 每步驟回呼 | 不強迫依賴特定追蹤系統，使用者可接任何工具 |
| **公開 API 收 `PIL.Image`** | 內部轉換為 Pydantic AI 格式 | 裁切、放大等視覺動作皆以 PIL 操作；使用者也慣用 PIL |

**影像格式轉換點（實作重點）**

Pydantic AI 的 `Agent.run()` **不接受 PIL Image**，其多模態輸入型別為：

```python
from pydantic_ai import Agent, BinaryContent, ImageUrl

result = await agent.run([
    prompt_text,
    BinaryContent(data=image_bytes, media_type='image/png'),
])
```

因此轉換責任歸屬明確：

| 層 | 影像型別 |
|---|---|
| 公開 API、視覺動作、幾何計算 | `PIL.Image.Image` |
| `VLMPort.ask()` 簽章 | `PIL.Image.Image`（保持核心一致） |
| `vlm/pydantic_ai.py` adapter 內部 | 轉為 `BinaryContent(data=bytes, media_type=...)` |

**adapter 是唯一知道 `BinaryContent` 存在的地方。** 此設計使日後更換 agent 框架時，僅需改寫該 adapter。

**模型字串格式**

Pydantic AI 的模型字串需帶 provider 前綴：

| Provider | 字串 | 環境變數 |
|---|---|---|
| Google Gemini | `google:gemini-2.5-flash` | `GOOGLE_API_KEY` |
| OpenAI | `openai:gpt-5.2` | `OPENAI_API_KEY` |
| Anthropic | `anthropic:claude-...` | `ANTHROPIC_API_KEY` |

OpenAI 相容端點（GLM、Qwen、OpenRouter、Ollama）需用類別而非字串：

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    'glm-4.6v',
    provider=OpenAIProvider(base_url='https://...', api_key=...),
)
```

### 4.4 例外階層

```python
class SaccadeError(Exception):
    """所有 Saccade 例外的基底，使用者可一次攔截。"""

class VLMError(SaccadeError):
    """VLM 呼叫失敗（網路、認證、額度）。"""

class ToolError(SaccadeError):
    """工具執行失敗。"""
```

**未收斂不拋例外**。達 `max_steps` 仍未達信心門檻時，回傳 `InvestigationResult(converged=False, ...)` 並附完整證據鏈。理由：未收斂為正常結果而非錯誤，且 BlindTest 需統計此類案例。例外僅用於真正的失敗。

### 4.5 相依與 extras

```toml
[project]
dependencies = [
    "pydantic>=2",
    "pydantic-ai>=1.0",
    "pillow>=10",
    "numpy>=1.26",
]

# 給使用者的 extras（pip install "saccade-vision[geometry]"）
[project.optional-dependencies]
geometry = ["opencv-python>=4.10"]
observability = ["langfuse>=2"]

# 開發依賴，不會被使用者安裝（uv 標準做法，PEP 735）
[dependency-groups]
dev = ["pytest", "pytest-asyncio", "pytest-cov", "ruff", "mypy", "pre-commit"]
```

**`optional-dependencies` 與 `dependency-groups` 的區別**：前者為使用者可選安裝的功能 extras，會發布至 PyPI metadata；後者為開發用依賴，不隨套件發布。開發工具放錯區塊會導致使用者安裝時被迫下載 pytest 等無關套件。

**opencv 列為選配**：體積逾 60MB，僅使用視野操作的使用者無需安裝。未安裝而呼叫幾何功能時，拋出明確訊息：`幾何驗證需要 opencv，請執行 pip install "saccade-vision[geometry]"`。

**Langfuse 列為選配**：函式庫不應強制使用者採用特定追蹤系統。內部僅使用標準 `logging.getLogger("saccade")`，不設定 handler。

### 4.6 版本策略

遵循 Semantic Versioning。

- `0.x` 期間：公開 API 可能變動，變動記於 CHANGELOG
- `1.0` 之後：MAJOR 才可 breaking，MINOR 僅新增，PATCH 僅修正
- `_` 開頭的內部模組任何變動皆為 PATCH

---

## 5. Sandlot Baseball 應用設計（Clean Architecture）

### 5.1 分層命名說明

Uncle Bob 於 2012 年原文定義四個概念層：Entities / Use Cases / Interface Adapters / Frameworks & Drivers。原文明確指出「層數不限於四層」，且**未規範資料夾命名**。

本專案採 Python Clean Architecture 社群常見的資料夾命名：

| 概念層（Uncle Bob） | 本專案資料夾 | 內容 |
|---|---|---|
| Entities | `domain/` | 資料模型與**規則本身**（角度怎麼算、什麼叫退步） |
| Use Cases | `application/` | **流程編排**（載入→抽骨架→算指標→存檔→回傳）與 Port 介面 |
| Interface Adapters | `interfaces/` | FastAPI、CLI |
| Frameworks & Drivers | `infrastructure/` | MediaPipe、YOLO、Postgres、Saccade 接線 |

**domain 與 application 的界線**：domain 放**業務規則**，application 放**業務流程**。application 不得包含領域知識，僅協調 domain 物件完成任務。

**與 ALL AI 的 `domains/` 無關**：後者指業務場景（receipt、lc），是不同概念，僅名稱相近。

### 5.2 目錄結構

```
apps/sandlot-baseball/
├── pyproject.toml
├── src/sandlot/
│   ├── domain/
│   │   ├── models.py          # Session, Metric, JointAngle
│   │   ├── kinematics.py      # 角度、動力鏈順序的計算規則
│   │   └── comparison.py      # 何謂「與上次不同」的規則
│   ├── application/
│   │   ├── ports/             # PosePort, DetectPort, SessionRepoPort, VectorPort
│   │   └── use_cases/
│   │       ├── analyze_swing.py
│   │       ├── analyze_pitch.py
│   │       └── compare_sessions.py
│   ├── infrastructure/
│   │   ├── vision/            # MediaPipe, YOLO
│   │   ├── persistence/       # SQLAlchemy + Postgres
│   │   ├── vector/            # pgvector
│   │   └── saccade_tools.py   # 將 MediaPipe/YOLO 包成 Saccade Tool
│   └── interfaces/
│       ├── api/               # FastAPI
│       └── cli/
├── frontend/                  # React + TS + Vite + Tailwind + shadcn/ui
├── knowledge/                 # RAG 原始文件
└── tests/
```

### 5.3 資料流

```
1. 上傳影片
2. [infrastructure] MediaPipe 逐幀 → 33 關節點 × N 幀
3. [domain] 品質規則判定 → 標記不可信幀（遮擋／模糊／可見度低）
4. [Saccade] 主動視覺介入
     不可信幀：裁切放大 → VLM 判斷遮擋原因 → 比對前後幀連續性 → 插值或捨棄
     關鍵事件幀：多幀取樣 → VLM 判定事件 → 與速度曲線極值比對
5. [domain] 以修正後骨架計算指標（純幾何，可驗算）
6. [infrastructure] YOLO 偵測球棒／球 → 補充揮棒平面、觸擊點
7. [application] 數字 + RAG → 解讀（強制附數字與出處）
8. [application] 與歷史 session 比對
9. 輸出：報告 + 證據鏈 + 標註影片
```

### 5.4 追問機制

分析完成後可追問，但**追問模式下 VLM 無「重新觀看影片」的工具**，僅能存取：本次分析數字、RAG 知識庫、該使用者歷史 session。

超出範圍的問題明確拒答。此限制為刻意設計：不限制則對話將退化為無根據的泛泛之談。

---

## 6. 棒球測量指標

參考來源：Driveline OpenBiomechanics Project 公開資料集，以及 baseball-cv（yasumorishima）的分析發現。

### 6.1 已知關鍵發現（決定測什麼）

**投球（60 位投手）**：
- 手臂速度相同（24–26 m/s）者，球速差距達 13 mph
- 跨步長度、抬腿彈性、手臂鏈模式、膝蓋平滑度共額外解釋 17.8% 變異
- 同手臂速度下，最佳 20% 與最差 20% 差 10.3 mph（89 vs 79 mph）

**打擊（40 位打者）**：
- 棒速幾乎無法預測擊球初速（R²=0.097）
- 重心轉移（跨步）為主導因素，額外解釋 37.8% 變異
- 效率最高 20% 的打者，棒速較慢卻多產生 20.9 mph 擊球初速

此發現構成產品價值主張：**教練普遍強調「揮快一點」，數據顯示重心轉移才是關鍵。**

### 6.2 測量項目與順序

| 階段 | 指標 | 計算方式 |
|---|---|---|
| **一致性**（M3 優先） | 上列各指標與前次 session 的差值 | 純數值比較，輸出「與上次差多少」而非「與參考值差多少」 |
| 打擊 | 髖肩分離角 | 肩連線向量 vs 髖連線向量夾角 |
| 打擊 | 重心轉移時機與幅度 | 關節加權質心軌跡 |
| 打擊 | 揮棒平面角 | YOLO 球棒軌跡擬合 |
| 投球 | 跨步長度 | 前後腳距離／身高標準化 |
| 投球 | 手肘外翻角 | 肩→肘→腕三點夾角 |
| 投球 | 動力鏈順序 | 各環節角速度峰值時序 |

**一致性優先的理由**：「你這次比上次少 4.8°」是純數學，不會錯；「你應該要 45°」需權威背書，錯了會被打臉。先站在不會錯的地基上。

### 6.3 RAG 知識庫來源

| 來源 | 內容 |
|---|---|
| Driveline Baseball | 開源程度最高的棒球運動科學 |
| ASMI | 投球受傷力學權威，論文免費 |
| arXiv / PubMed | pitching biomechanics、swing kinematics |
| 中華棒協／大學運動科學系教材 | 中文內容 |

**授權注意**：baseball-cv 為 CC BY-NC-SA 4.0（非商業）。本專案僅參考其「哪些指標重要」的知識性發現（不受著作權保護），不複製其程式碼。README 中致謝引用。

---

## 7. 技術選型

| 項目 | 選擇 | 理由 |
|---|---|---|
| 語言 | Python 3.11+ | `Self`、`TypeVarTuple` 等型別功能 |
| 套件管理 | **uv + uv.lock** | 2026 標準，取代 pip+venv+poetry，快 10–100 倍，有 lockfile |
| Lint / Format | Ruff | 取代 black+flake8+isort |
| 型別檢查 | mypy | 成熟標準（`ty` 尚在 preview，未採用） |
| Agent 框架 | Pydantic AI | 結構化輸出、工具、重試；適合線性工具呼叫工作流 |
| VLM | Gemini Flash 主力（可換） | 免費層 1500 req/day，支援圖片，免信用卡 |
| 姿態估計 | MediaPipe Pose | CPU 可跑、33 關節點、免費 |
| 物件偵測 | YOLOv8n 官方權重 | COCO 已含 `baseball bat`、`sports ball`，**無需訓練** |
| 影像處理 | OpenCV + Pillow + NumPy | 標準 |
| 資料庫 | Supabase（純 Postgres） | SQLAlchemy 2.0 直連 |
| ORM / Migration | SQLAlchemy 2.0 + Alembic | 標準 |
| 向量 | Supabase pgvector | `pgvector` 套件 + SQLAlchemy 整合 |
| API | FastAPI | 標準 |
| 前端 | React + TypeScript + Vite + Tailwind + shadcn/ui | 2026 最主流組合 |
| 測試 | pytest + pytest-asyncio | 標準 |
| Git hook | pre-commit | 提交時擋下違規 |
| CI | GitHub Actions | 標準 |
| Monorepo | **uv workspace** | 官方支援，拆分時無痛 |

**Supabase 使用規範（硬性）**：僅使用 Postgres 與 pgvector，以 SQLAlchemy 直連 `DATABASE_URL`。**禁止使用 supabase-py SDK、Auth、Storage、Realtime、Edge Functions。** 確保日後可遷移至任何 Postgres 實例。

### 7.1 成本控制

**Gemini 免費層**：1500 requests/day，1M tokens/min，免信用卡。

| 情境 | 呼叫數 | 免費層可行性 |
|---|---|---|
| 一支棒球影片 | ~20 | 每日可跑 75 支 |
| BlindTest 一輪 | ~4000 | 分 3 日跑完 |

其他免費選項：Groq（30 req/min）、OpenRouter（28+ 免費模型含視覺）、Cerebras、NVIDIA NIM。

**分級路由**：簡單視覺判斷用 Flash，複雜仲裁用 Pro。分界由 M2 消融實驗數據決定。

**結論：本專案可於零 API 成本下完成。**

---

## 8. Repo 策略

### 8.1 階段

| 階段 | 結構 |
|---|---|
| M0–M2 | 僅 `saccade` repo（引擎 + benchmark + examples） |
| M3–M5 | 同 repo，`apps/sandlot-baseball/`，以 **uv workspace** 管理 |
| M6 | `apps/sandlot-baseball/` 搬出為獨立 repo，改依賴 PyPI 版本 |

**uv workspace 設定**：

```toml
# 根 pyproject.toml（saccade 引擎本身）
[tool.uv.workspace]
members = ["apps/*"]
```

```toml
# apps/sandlot-baseball/pyproject.toml
[project]
name = "sandlot-baseball"
dependencies = ["saccade-vision", "mediapipe", "ultralytics", ...]

[tool.uv.sources]
saccade-vision = { workspace = true }    # 指向本機 workspace 而非 PyPI
```

兩套件共存且各有 `pyproject.toml`，應用可依賴本機路徑的引擎而無需發版。**拆分時僅需搬移資料夾並刪除 `[tool.uv.sources]` 區塊**，依賴自動改為解析 PyPI 上的正式版本。

**拆分判準**（看狀態非看時間）：連續兩週開發應用功能未需修改引擎 → 引擎介面已穩定。

### 8.2 一個應用一個 repo

未來若有其他應用（如籃球、PCB 檢測），一律各自開 repo，不併入 `saccade`。理由：依賴衝突、版本綁死、README 失焦、star 分散。

`examples/` 僅放 20–50 行的示範 script，不放完整應用。

### 8.3 命名

| 項目 | 名稱 | 備註 |
|---|---|---|
| 引擎 GitHub repo | `saccade` | |
| 引擎 PyPI 套件 | `saccade-vision` | `saccade` 已被註冊（HTTP 200 實測） |
| 引擎 import 名 | `saccade` | pip 名與 import 名不同屬常見做法（`scikit-learn`/`sklearn`） |
| 應用 repo | `sandlot-baseball` | |
| 本機資料夾 | `Saccade` | |

---

## 9. 里程碑與驗收

| 里程碑 | 內容 | 驗收條件 | 估時 |
|---|---|---|---|
| **M0** | repo 骨架、公開 API 型別定義、CI、pre-commit、測試框架 | `uv run pytest` 全綠；`uv run mypy src` 無錯；CI 綠；`import saccade` 可取得所有 `__all__` 名稱 | 1 週 |
| **M1** | Perceive-Verify Loop 最小版 + 3 動作（裁切／放大／幾何量測）+ 快取 | **跑出 BlindTest 第一個數字**（基準 58%）；FakeVLM 測試全綠 | 2 週 |
| **M2** | BlindTest 完整（7 任務 × 多模型）+ 消融實驗 | **結果表進 README**；消融數據說明各策略貢獻 | 1 週 |
| **M3** | Sandlot 應用骨架 + MediaPipe + YOLO + 指標計算 + **動作一致性** | 同一影片跑十次數字完全相同；兩支影片能算出差異 | 2 週 |
| **M4** | Saccade × 棒球：以主動視覺修正遮擋／模糊幀 | **量化證明**：有／無 agent 的可用幀數與事件偵測準確率差異 | 2 週 |
| **M5** | RAG 知識庫 + 解讀 | 抽查 20 句建議，**0 句無數字或無出處** | 2 週 |
| **M6** | 前端 + **拆分 repo** + 引擎發布 PyPI | 全流程可跑；`pip install saccade-vision` 成功；demo 影片完成 | 2 週 |
| **M7** | README、Docker、文件 | 他人可依 README 從零跑起 | 1 週 |

**總計約 13 週**（兼職 3–4 個月）。

**提前可交付點**：M0–M2 完成（約 4 週）即為完整可放履歷的成果——引擎 + 公開 benchmark 數字。應用為加分項。

---

## 10. 風險

### 10.1 已驗證可行

| 項目 | 依據 |
|---|---|
| MediaPipe 做棒球分析 | baseball-cv 已實作，論文佐證 |
| YOLO 官方權重認球棒／球 | COCO 類別表確認 |
| BlindTest 為公開 benchmark | 程式碼公開，58% 基準明確 |
| 裁切放大可提升 VLM 準確率 | Agent0-VL、VipAct 等已證實 |
| Gemini 免費層支援視覺 | 官方文件 |
| uv workspace | uv 官方功能 |

### 10.2 有風險

| 風險 | 影響 | 對策 |
|---|---|---|
| BlindTest 提升幅度有限 | 核心數字不亮眼 | 消融實驗說明「哪些策略有效／無效／為何」，此本身即為發現。58%→68% 亦具意義 |
| 快速揮棒動態模糊 | 關鍵幀骨架失準 | ①範圍收在較慢階段 ②要求 60fps ③**主動視覺偵測「此幀不可信」本身即為賣點** |
| YOLO 認不出模糊球棒 | 揮棒平面無法計算 | 備案：VLM 標註關鍵幀／光流追蹤／先只做人體 |
| Gemini Flash 視覺能力不足 | 需改付費模型 | 分級路由；M2 提供決策數據 |
| RAG 知識庫建置費時 | M5 延宕 | 一致性分析不需知識庫，可先交付 |

### 10.3 未知數

| 未知數 | 確認時點 |
|---|---|
| 主動視覺對棒球骨架修正的實際幫助 | M4 |
| 手機拍攝影片品質是否足夠 | M3 第一天 |
| 引擎公開 API 設計是否正確 | M3–M5（被應用使用後） |

---

## 11. 防技術債硬規則

以下寫入 `CLAUDE.md`，違反視為缺陷。

1. **測量與解釋分離**。數值由可計算工具產生，VLM 僅驗證與解釋。
2. **Saccade 不得 import 領域套件**（MediaPipe、YOLO 等）。領域能力一律經 `register_tool()` 注入。
3. **Saccade 純邏輯模組不得做檔案 I/O**。`_planner`、`_verifier`、`geometry/` 不呼叫 `cv2.imread`、`Image.open` 等；影像載入由呼叫方負責，函式庫只接收記憶體物件。
4. **公開 API 變更須走 SemVer**。`_` 開頭模組可自由變動；`__all__` 內的名稱變動需記 CHANGELOG。
5. **新增參數一律 keyword-only**。避免破壞既有呼叫。
6. **VLM 快取自 M1 起實作**。省成本且確保測試可重現。
7. **測量規則 100% 單元測試覆蓋**。純數學，無藉口。
8. **輸出強制附證據**。任何建議須帶數字、幀號、資料來源。
9. **Supabase 僅當 Postgres/pgvector 使用**。
10. **依賴變更須更新 `uv.lock` 並提交**。
11. **每個里程碑通過驗收才進下一個**。

上述 2、3 由 `tests/test_architecture.py` 以 AST 掃描自動強制執行（檢查 import 與 I/O 函式呼叫）。

---

## 12. 參考資料

**論文**
- Rahmanzadehgervi et al., *Vision Language Models Are Blind* — https://vlmsareblind.github.io/
- *Position: The Systemic Lack of Agency in Visual Reasoning* (2026) — https://arxiv.org/pdf/2606.14795
- *QuantiPhy* — https://arxiv.org/pdf/2512.19526
- *Agent0-VL* — https://arxiv.org/html/2511.19900
- *Scalable Injury-Risk Screening in Baseball Pitching From Broadcast Video* — https://arxiv.org/pdf/2603.04864

**程式碼／資料集**
- BlindTest — https://github.com/anguyen8/vision-llms-are-blind
- baseball-cv（指標參考，CC BY-NC-SA 4.0）— https://github.com/yasumorishima/baseball-cv
- Driveline OpenBiomechanics Project

**架構與工程實務**
- Alistair Cockburn, *Hexagonal Architecture (Ports and Adapters)*, 2005
- Robert C. Martin, *The Clean Architecture*, 2012 — https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html
- Ben Hoyt, *Designing Pythonic library APIs* — https://benhoyt.com/writings/python-api-design/
- *Public API surface* — Real Python
- Seth Larson, *Designing Libraries for Async and Sync I/O*
- Semantic Versioning 2.0.0 — https://semver.org/
