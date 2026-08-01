# CLAUDE.md — Saccade 專案規則

本檔為 `docs/specs/2026-07-29-saccade-design.md` 第 11 節的落地版本。
**違反以下任一條視為缺陷（defect），不是風格建議。**

---

## 硬規則（11 條）

### 1. 測量與解釋分離
數值一律由可計算工具產生，VLM 僅負責驗證數字合理性與翻譯成人話。

```
❌ 影片 → VLM → 「你重心偏前」
✅ 影片 → 幾何計算 → 「重心於觸擊前 0.12s 越過前腳踝 3.2cm」 → VLM 解釋
```

判準：任何進入輸出的數字，必須能指出是哪個函式算出來的。VLM 產生的數字不算數。

### 2. Saccade 不得 import 領域套件
`src/saccade/` 底下禁止出現 `mediapipe`、`ultralytics`、`sklearn`、`torch`、`tensorflow` 等領域專用套件。
領域能力一律由應用透過 `register_tool()` 注入。

違反後果：引擎失去通用性，BlindTest 無法與應用共用同一核心。
自動強制：`tests/test_architecture.py`（AST 掃描 import）。

### 3. 純邏輯模組不得做檔案 I/O
`_planner.py`、`_verifier.py`、`_evidence.py`、`geometry/` 不得呼叫
`cv2.imread`、`cv2.imwrite`、`cv2.VideoCapture`、`Image.open`、`Image.save`、`open`。

影像載入由呼叫方負責，函式庫只接收記憶體物件（`PIL.Image.Image`、`np.ndarray`）。
自動強制：`tests/test_architecture.py`（AST 掃描 call）。

### 4. 公開 API 變更須走 SemVer
- `_` 開頭的內部模組：任何變動皆為 PATCH，可自由修改
- `__all__` 內的名稱：變動須記入 `CHANGELOG.md`
- `0.x` 期間公開 API 可變動但仍須記錄；`1.0` 後 MAJOR 才可 breaking

### 5. 新增參數一律 keyword-only
公開函式與方法的新參數必須放在 `*` 之後。

```python
def investigate(self, image, question, *, expect=None) -> InvestigationResult: ...
```

理由：日後新增參數不破壞既有 positional 呼叫。

### 6. VLM 快取自 M1 起實作
所有 VLM 呼叫走 `CachePort`。鍵 = `sha256(圖片 bytes + prompt + model_id)`。
理由：省 API 成本，且確保測試可重現。

### 7. 測量規則 100% 單元測試覆蓋
`geometry/`、`domain/kinematics.py`、`domain/comparison.py` 等純數學模組需 100% 行覆蓋。
每個函式至少三案例：正常／邊界／退化（例：兩圓「剛好相切」必測）。

### 8. 輸出強制附證據
任何建議、結論、答案須帶：數字 + 幀號／座標 + 資料來源。
`InvestigationResult` 一律附完整 `evidence_chain`。無證據的句子不得輸出。

### 9. Supabase 僅當 Postgres/pgvector 使用
以 SQLAlchemy 2.0 直連 `DATABASE_URL`。
**禁止** supabase-py SDK、Auth、Storage、Realtime、Edge Functions。
理由：確保可遷移至任何 Postgres 實例。

### 10. 依賴變更須更新 `uv.lock` 並提交
`uv.lock` 進 git，不得 ignore。改 `pyproject.toml` 依賴後跑 `uv sync` 並把 lock 一起 commit。

### 11. 每個里程碑通過驗收才進下一個
驗收條件寫在各 `docs/plans/M*.md`。
必須**貼出實際指令輸出**，不得只宣稱「通過」。未過不得開始下一個里程碑。

---

## 未收斂不是錯誤

達 `max_steps` 仍未達信心門檻時，回傳 `InvestigationResult(converged=False, ...)` 附完整證據鏈，
**不拋例外**。例外僅用於真正的失敗（VLM 網路錯誤、工具崩潰）。
理由：未收斂為正常結果，且 BlindTest 需統計此類案例。

---

## 常用指令

```bash
uv sync --all-packages --all-extras --dev   # 同步環境（--all-packages 才會裝 apps/）
uv run pytest                               # 測試（含 apps/ 的測試）
uv run ruff check src tests benchmarks apps # lint
uv run ruff format .                        # format
uv run mypy src apps/sandlot-baseball/src   # 型別檢查
uv build                                    # 建置
uv run pre-commit run --all-files
```

`--all-packages` 不是選配。這是 uv workspace，少了它 `apps/` 底下的應用不會被安裝，
連帶 MediaPipe 與 YOLO 也不會 —— 而測試會照樣通過，因為那些測試根本沒被收集到。

---

## 架構速查

| 產出 | 架構風格 | 位置 |
|---|---|---|
| Saccade（引擎，library） | 扁平 + 明確公開 API | `src/saccade/` |
| Sandlot Baseball（應用） | Clean Architecture | `apps/sandlot-baseball/`（M3 起） |

命名慣例：`_module.py` 為內部實作，無底線者為公開契約。

工具三類，`is_measurement=True` 者才能對質 VLM：
- 視野操作（crop / zoom / annotate）— 無 measurement
- 計算工具（measure_geometry / count_contours）— `True`
- 領域工具（應用註冊）— 由註冊者宣告

**核心心法：工具是 VLM 的裁判，不是 VLM 的延伸。**
