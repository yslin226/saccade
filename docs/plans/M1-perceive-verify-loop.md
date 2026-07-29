# M1 — Perceive-Verify Loop 最小版

**目標**：讓 `ActiveVisionAgent.investigate()` 真的會跑，並**跑出 BlindTest 的第一個數字**。
**前置**：M0 完成（已驗收，commit `5a418ca`）。
**估時**：2 週
**參考**：`docs/specs/2026-07-29-saccade-design.md` 第 3 節、4.3 節

---

## 這個里程碑要證明什麼

M0 證明了骨架站得住。M1 要回答的是唯一重要的問題：

> **讓 VLM 自己決定看哪裡，準確率會不會變高？**

答案是多少都可以接受（含「沒有提升」），**但必須是可重現的真實數字**。
不得預告、不得估計、不得挑好看的子集。

---

## 驗收條件（全部通過才進 M2）

```bash
uv run pytest                    # 全綠，且新增的 loop 測試不依賴真實 API
uv run ruff check src tests
uv run ruff format --check .
uv run mypy src
uv run pytest --cov=saccade --cov-report=term-missing   # geometry/ 維持 100%
```

外加：
1. **BlindTest 至少一個任務跑出數字**，並記錄：
   - baseline（VLM 直接回答，無 agent）
   - saccade（走完整 loop）
   - 樣本數、模型名稱、日期
2. 同一份輸入**跑兩次結果完全相同**（快取 + 固定 seed）
3. `examples/` 有一支 20–50 行、不需 API key 就能跑的 FakeVLM 範例
4. CI 綠燈

**注意**：驗收 1 不要求 saccade > baseline。要求的是數字為真。

---

## Task 1：VLM adapter（Pydantic AI）

**檔案**：`src/saccade/vlm/pydantic_ai.py`

實作 `VLMPort`。**此為整個專案唯一知道 `BinaryContent` 存在的地方**（spec 4.3）。

```python
class PydanticAIVLM:
    def __init__(self, model: str | Model, *, system_prompt: str | None = None): ...

    @property
    def model_id(self) -> str: ...

    async def ask(self, images, prompt, output_type=None) -> VLMResponse: ...
```

**要點**：
- `PIL.Image.Image` → `BinaryContent(data=png_bytes, media_type="image/png")`，轉換只在此檔
- `output_type` 有值時走 Pydantic AI 的結構化輸出，填入 `VLMResponse.structured`
- 從 usage 取 token 數填 `tokens_used`
- 網路／認證／額度錯誤一律包成 `VLMError`（附原始例外為 `__cause__`）

**測試**：mock Pydantic AI 的 `Agent.run`，斷言轉換正確、錯誤有包裝。**不打真實 API**。

---

## Task 2：三個視覺動作

**檔案**：`src/saccade/actions/crop.py`、`zoom.py`、`annotate.py`

| 動作 | 簽章 | 說明 |
|---|---|---|
| `crop` | `(image, bbox) -> tuple[Image, Viewport]` | 回傳新圖與對應 viewport |
| `zoom` | `(image, bbox, factor) -> tuple[Image, Viewport]` | 裁切後放大，用 LANCZOS |
| `annotate` | `(image, boxes, labels) -> Image` | 畫框與標籤，供證據鏈存證 |

**硬規則**：
- 全部**純函式**，不改動輸入影像（`.copy()` 後再操作）
- 不做檔案 I/O（規則 3）
- 座標一律能映射回原圖 —— `Viewport.source_size` 永遠指原圖尺寸，不是裁切後尺寸

**測試**：裁切後尺寸正確、放大後 viewport 座標仍指向原圖、輸入影像未被改動。

---

## Task 3：`_observer.py` — 呼叫 VLM

**檔案**：`src/saccade/_observer.py`

職責：組 prompt、呼叫 `VLMPort`、**先查快取**、回傳 `Observation`。

- 快取鍵用 M0 的 `make_cache_key()`（規則 6）
- 命中快取不計入 `total_tokens`
- prompt 要求 VLM 輸出結構化陳述與自評信心

**測試**：FakeVLM + MemoryCache。斷言第二次相同呼叫**沒有再打 VLM**（`call_count` 不變）。

---

## Task 4：`_verifier.py` — 衝突仲裁

**檔案**：`src/saccade/_verifier.py`

**這是本專案的本體，不是普通 agent 的 tool call。**

```python
def verify(observation: Observation, results: list[ToolResult]) -> Verification: ...
```

規則：
- **只有 `is_measurement=True` 的結果可以當裁判**（spec 3.3）。其餘一律忽略，不得用來提高信心
- 一致 → `passed=True`，信心上調
- 衝突 → `passed=False`，寫入 `conflict` 字串（要具體：「VLM 說 3 個人，偵測到 2 個」）
- 無可用測量 → `passed` 依 VLM 自評，但信心**不得**超過未驗證上限（建議 0.6）

**純邏輯，不做 I/O**（規則 3，架構守衛會掃）。

**測試**：一致／衝突／無測量／只有非測量結果 四種情況，各自的信心變化。
特別測「只給 `is_measurement=False` 的結果時，信心不會上升」——這條錯了整個設計就垮了。

---

## Task 5：`_planner.py` — 決定下一步

**檔案**：`src/saccade/_planner.py`

```python
def plan_next(question, viewport, evidence, confidence) -> PlannedAction: ...
```

M1 用**規則式**即可，不必讓 VLM 決定（先讓 loop 跑起來）：
1. 第一步：全圖
2. 有衝突：換角度（放大衝突區域）
3. 信心低且未探索完：掃描未探索區域
4. 連續兩步無新資訊：停

需維護**已探索區域集合**，避免來回看同一塊。

**純邏輯，不做 I/O**。

**測試**：給定不同 evidence 狀態，斷言選出的動作符合預期；斷言不會重複探索同一區域。

---

## Task 6：`_evidence.py` 與 loop 組裝

**檔案**：`src/saccade/_evidence.py`、`src/saccade/agent.py`

`_evidence.py`：累積 `EvidenceStep`，`image_ref` 只存參考字串（規則 3，不寫檔）。

`agent.py`：把 Plan → Act → Observe → Verify → Record 串起來，實作 M0 定好的簽章。

**停止條件**（三者取先到）：
- `confidence >= confidence_threshold`
- `len(evidence) >= max_steps`
- planner 回報無新資訊

**未收斂回傳 `converged=False`，不拋例外**（spec 4.4）。

`investigate()` 同步版包 async：注意已在 event loop 中時要給明確錯誤，不要 deadlock。

**測試**：FakeVLM 腳本化多步回應，斷言
- 收斂案例步數正確
- 未收斂案例 `converged=False` 且證據鏈完整
- `on_step` callback 每步都被呼叫
- 全程零真實 API 呼叫

---

## Task 7：BlindTest 執行器

**檔案**：`benchmarks/blindtest/`（不放進 `src/`，不隨套件發布）

1. 取得 BlindTest 資料（https://github.com/anguyen8/vision-llms-are-blind）
2. 先做**兩個任務**：`circles_overlap`、`line_intersections`
   （這兩個 M0 的 geometry 已有對應的可計算裁判）
3. 兩種模式跑：`--mode baseline`（直接問）、`--mode saccade`（走 loop）
4. 輸出 JSON：每題的答案、正確與否、步數、token 數、證據鏈

**成本控制**：全程走快取。Gemini 免費層 1500 req/day，先跑小樣本（每任務 50 題）確認管線，再擴大。

**測試**：用 FakeVLM 跑一次完整管線，確認評分邏輯正確——**評分器本身不能有 bug**，否則數字沒有意義。

---

## Task 8：examples

**檔案**：`examples/minimal.py`

20–50 行，用 FakeVLM，**不需 API key**。示範：建 agent → investigate → 印出證據鏈。

同時作為 README 的可用範例（M0 README 尚未有範例，M1 補上）。

---

## 完成後

1. 跑一次全部驗收條件，**貼出實際輸出**
2. 把 BlindTest 數字寫進 README 的「現在的狀態」——**照實寫，不美化**
3. commit + push，確認 CI 綠
4. **產出 M2 plan**，再繼續

---

## M1 不做的事

- 完整 7 個 BlindTest 任務（M2）
- 多模型比較、消融實驗（M2）
- 讓 VLM 自己規劃下一步（M1 規則式就夠，之後再看要不要換）
- 任何棒球相關（M3）

---

## 風險

| 風險 | 對策 |
|---|---|
| 提升幅度為零甚至變差 | **照實寫**。M2 的消融實驗解釋為什麼，這本身就是發現 |
| 免費層額度不夠 | 快取 + 小樣本先驗證管線；跨日分批跑 |
| 規則式 planner 太笨 | M1 只要 loop 會動；planner 好壞是 M2 消融的題目 |
