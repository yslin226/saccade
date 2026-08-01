# M3 — Sandlot Baseball 骨架與動作一致性

**目標**:建立應用骨架,接上 MediaPipe 與 YOLO,算出棒球指標,並證明同一支影片重複分析結果完全相同。
**前置**:M2 完成(引擎可用,工具選擇與對質機制已驗證,commit `897a346`)。
**估時**:2 週
**參考**:`docs/specs/2026-07-29-saccade-design.md` 第 5、6、8 節

---

## 第 0 天:地基測試(已完成)

驗收條件寫「同一影片跑十次數字完全相同」。**在寫任何程式之前必須先確認偵測器允許這件事** ——
MediaPipe 與 Ultralytics 都沒有承諾 bitwise 可重現性,而每個指標都建在它們的座標上。
地基會動,「你這次比上次少 4.8°」就變成在描述偵測雜訊。

實測 `happy/林永閎.MOV`,12 幀均勻取樣,1920×1080:

```
=== 同一 process,5 次 ===
  MediaPipe   bitwise identical across 5 runs: True
  YOLO11-pose bitwise identical across 5 runs: True

=== 獨立 process,5 次 ===
  mediapipe=1e94edf3ee6360162f634472   ×5 全同
  yolo=129dac9e856dc1a035647056        ×5 全同
```

**結論:可以達成,且不需額外約束。** 不用固定 seed、不用強制 CPU-only、不用量化座標。

### 但這個結果綁定兩個條件,兩者都要寫成程式碼的契約

| 條件 | 為什麼 |
|---|---|
| `RunningMode.IMAGE` | 不是 `VIDEO`。VIDEO 模式帶跨幀狀態,從不同幀開始會得到不同結果 |
| 解碼與偵測分離 | 先抽幀成 array 再偵測。若在每次執行中重新解碼,測到的是「解碼器 + 偵測器」,而解碼器不穩的症狀看起來會像偵測器的問題 |

### 未測:跨機器

實測只在同一台機器。MediaPipe 的 XNNPACK 依 CPU 指令集選不同 kernel,不同機器**很可能**不同。
驗收條件不要求跨機器,所以不擋 M3,但「一致性」若要用於跨裝置比較,那是另一個問題。

版本:mediapipe 1.0.0、ultralytics 8.4.113、opencv 5.0.0、numpy 2.4.6。
**版本變動即可能改變結果**,所以 session 記錄必須存下版本字串。

---

## 這是骨架,不是原型

目錄結構即最終結構,M3 只是填一部分。不會重寫:`domain/kinematics.py` 算出的髖肩分離角,
M6 上網頁時是同一個函式。

| 層 | M3 | 之後 |
|---|---|---|
| `domain/` | ✅ 全部 | |
| `application/ports/` | ✅ 介面 | |
| `application/use_cases/` | ✅ analyze_pitch, compare_sessions | |
| `infrastructure/vision/` | ✅ MediaPipe + YOLO | |
| `infrastructure/persistence/` | ⚠️ JSON 檔 | M5 換 Postgres |
| `infrastructure/vector/` | ❌ | M5 (RAG) |
| `interfaces/cli/` | ✅ | |
| `interfaces/api/` | ❌ | M6 |
| `frontend/` | ❌ | M6 |

**persistence 先用 JSON**:`SessionRepoPort` 的介面照 Postgres 的形狀設計(id、時間戳、查詢),
M5 建 RAG 時本來就要開資料庫,一起換。規則 9 規範的是「用 Postgres 時只當 Postgres 用」,
不是「任何時候都得有資料庫」。Port 定好之後換實作很便宜,那正是 Clean Architecture 的用意。

---

## 驗收條件

```bash
uv run pytest
uv run ruff check src tests benchmarks apps
uv run ruff format --check .
uv run mypy src apps/sandlot-baseball/src
```

外加,必須貼出實際輸出:

1. **同一支影片分析十次,數字完全相同**
   ```bash
   uv run sandlot analyze happy/林永閎.MOV --repeat 10
   ```
   十次的指標雜湊必須全部相同。

2. **兩支影片能算出差異**
   ```bash
   uv run sandlot compare <session-a> <session-b>
   ```
   輸出每個指標的差值,附幀號與座標(規則 8)。

3. **`domain/kinematics.py` 與 `domain/comparison.py` 100% 行覆蓋**(規則 7)
   每個函式至少三案例:正常/邊界/退化。

4. **架構守衛通過** —— app 可以 import mediapipe,引擎不行。
   現有的 `tests/test_architecture.py` 只掃 `src/saccade`,需確認 app 加入後仍只掃引擎。

---

## Task 1 — workspace 與骨架

根 `pyproject.toml` 加:

```toml
[tool.uv.workspace]
members = ["apps/*"]
```

`apps/sandlot-baseball/pyproject.toml`:

```toml
[project]
name = "sandlot-baseball"
dependencies = ["saccade-vision", "mediapipe", "ultralytics", "opencv-python"]

[tool.uv.sources]
saccade-vision = { workspace = true }
```

M6 拆分時只需搬資料夾並刪掉 `[tool.uv.sources]`,依賴自動改為解析 PyPI。

**驗收**:`uv sync` 成功,`uv run python -c "import sandlot"` 可執行,引擎的架構守衛仍綠。

---

## Task 2 — domain:指標怎麼算

純數學,不碰檔案、不碰 MediaPipe。輸入是座標,輸出是數字。

`domain/models.py`
- `JointReading` — 一個關節在一幀的位置與信心
- `Frame` — 一幀的所有關節 + 幀號 + 時間戳
- `Metric` — 一個指標的值、單位、以及**算它用到的幀號與座標**(規則 8)
- `Session` — 一次分析的完整結果 + 版本字串 + 影片雜湊

`domain/kinematics.py`(引擎的 `saccade.geometry` 提供原語,這裡是棒球規則)
- `hip_shoulder_separation` — 肩連線 vs 髖連線夾角
- `elbow_valgus` — 肩→肘→腕三點夾角
- `stride_length` — 前後腳距離 / 身高標準化
- `centre_of_mass_path` — 加權質心軌跡
- `kinetic_chain_order` — 各環節角速度峰值時序

`domain/comparison.py`
- `difference(a: Session, b: Session) -> list[MetricDelta]`
- 純數值比較,輸出「與上次差多少」而非「與參考值差多少」

**為什麼一致性優先**(spec 6.2):「你這次比上次少 4.8°」是純數學不會錯;
「你應該要 45°」需要權威背書,錯了會被打臉。先站在不會錯的地基上。

**驗收**:100% 行覆蓋,每個函式三案例。退化案例明確 raise 而非回傳看似合理的數字 ——
這是 `saccade.geometry` 已經確立的慣例:`angle_between` 遇到零長度射線會 raise,
因為回傳 0.0 會通過門檻比較然後安靜地回答錯的問題。

---

## Task 3 — application:流程與 Port

`application/ports/`
- `PosePort` — `detect(frames) -> list[Frame]`
- `DetectPort` — YOLO 物件偵測(球棒、球)
- `SessionRepoPort` — `save(session)`, `get(id)`, `list_for(user)`

Port 介面照 Postgres 的形狀設計,即使 M3 的實作是 JSON 檔。

`application/use_cases/analyze_pitch.py`
- 載入影片 → 抽幀 → PosePort → domain 算指標 → 存檔 → 回傳 Session
- **不含領域知識**,只協調 domain 物件(spec 5.1)

`application/use_cases/compare_sessions.py`
- 取兩個 Session → `domain.comparison.difference` → 回傳差值

**驗收**:use case 的測試用假的 Port,不碰 MediaPipe 也不碰檔案系統。

---

## Task 4 — infrastructure:接上偵測器

`infrastructure/vision/mediapipe_pose.py`
- 實作 `PosePort`
- **必須 `RunningMode.IMAGE`** —— 第 0 天的測試綁定這個條件,寫成程式碼的常數與註解
- 解碼與偵測分離:先 `sample_frames` 成 array,再逐幀 detect

`infrastructure/vision/yolo_detect.py`
- 實作 `DetectPort`

`infrastructure/persistence/json_repo.py`
- 實作 `SessionRepoPort`,一個 session 一個 JSON 檔
- 存版本字串與影片雜湊 —— 版本變動會改變結果,不記就無法解釋差異
- 存放位置由建構參數決定,CLI 以 `--data-dir` 傳入,預設 `~/.sandlot/sessions/`。
  測試一律傳 `tmp_path`,所以測試不可能寫進真實家目錄 —— 與 `FileCache` 同一慣例

`infrastructure/saccade_tools.py`
- 把 MediaPipe/YOLO 包成 `saccade.Tool`,經 `register_tool()` 注入
- 兩偵測器的不一致工具已存在(`benchmarks/pose_probe/disagreement.py`,AUROC 0.638),
  M4 才真正用它,M3 先把接線做好

**驗收**:同一影片十次,指標雜湊全同。

---

## Task 5 — CLI

`interfaces/cli/`
- `sandlot analyze <video> [--repeat N]`
- `sandlot compare <session-a> <session-b>`

輸出必須帶幀號與座標(規則 8)。無證據的句子不得輸出。

---

## 已知風險

| 風險 | 對策 |
|---|---|
| 手機影片畫質不足以偵測 | 第 0 天已確認 `林永閎.MOV` 可偵測到骨架。若其他影片不行,是資料問題不是程式問題 |
| 快速揮棒動態模糊 | M4 處理。M3 只需要指標算得出來且可重現 |
| YOLO 認不出模糊球棒 | 揮棒平面指標可能無法計算。備案:M3 只做人體指標,球棒延後 |
| 版本升級改變數字 | session 存版本字串。跨版本比較時明確拒絕,而非給出無意義的差值 |
| 跨機器不一致 | 未測。M3 不要求,但若之後要跨裝置比較需重新評估 |

---

## 不做

FastAPI、Postgres、pgvector、frontend、RAG、追問機制。
這些在 M5–M6,現在做等於在需求還沒被使用驗證前先蓋。
