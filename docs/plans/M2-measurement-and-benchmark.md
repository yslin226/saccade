# M2 — 讓 Verifier 有東西可驗,然後量真的

**目標**:接上可計算的測量工具,讓 Perceive-Verify Loop 的驗證環節真的運作,再跑完整 BlindTest 與消融實驗。
**前置**:M1 完成(loop 可跑、benchmark 管線可用、commit `0e586c7`)。
**估時**:1–2 週
**參考**:`docs/specs/2026-07-29-saccade-design.md` 第 3.3、11 節

---

## M1 留下的問題(M2 的起點)

M1 跑出數字了,但那個數字量錯東西。實測證據(`benchmarks/blindtest/results/`):

```
gpt-5.4, 150 題, Touching Circles
  baseline 90.7%  →  saccade 93.3%   (+2.7%, McNemar p=0.388 不顯著)
  converged: 0/150
  distinct confidence values: [0.30]
```

**150 題信心值全部相同、零收斂。** 原因:沒有註冊任何 `is_measurement=True` 的工具,verifier 每次回傳 `method="none"`,信心卡在 `UNVERIFIED_CEILING`。

所以 M1 跑的是「看三次取最後一次」,不是 Perceive-Verify Loop。**spec 1.4 的核心主張(測量與解釋分離)從未被測試過。**

### 補充:第二個原因(M1 收尾 review 才發現,commit `428d74d`)

原本以為「沒收斂」只是因為沒註冊工具。review 程式碼後發現**還有一個更根本的原因**:

`_boolean_conflict` 要求測量的欄位名出現在句子裡才判斷。但 benchmark 要的答案就是 `Yes` / `No`,`"overlap"` 這個字根本不會出現:

```
'No'  vs 測量 overlap=True   →  passed=True   ← 應該衝突
'Yes' vs 測量 overlap=False  →  passed=True   ← 應該衝突
```

**裁判一次都沒推翻過模型。** 意思是:就算 M1 有註冊工具,verifier 一樣不會 fire —— 而症狀會看起來像偵測工具的問題,查錯方向。

已修(`_verdict()` 讀開頭的 yes/no,取最先出現者),並補了 `TestBareVerdicts` 迴歸測試。

**這件事對 M2 的意義**:
1. Task 3 的「確認 verifier 真的在動」不是形式檢查,是**已知有過失效前例**的關卡
2. 測試綠燈不代表功能正確 —— M1 全程 432 tests 綠,但驗證環節從未運作。M2 每個工具接上後,**必須用真實模型輸出的字串驗證**,不能只用自己想像的句子
3. `_verifier.py` 其餘的比對邏輯(`_count_conflict`)同樣只在想像的輸入上測過,Task 3 要一併驗

再看細節:8 個修對的全是 `gt=No`,4 個弄壞的有 3 個把 `Yes` 改成 `No`。像是放大讓模型偏向答 No,不是看得更準。

**M2 要回答的是 M1 沒能回答的那個問題:當工具真的能對質 VLM 時,準確率會不會提升。**

---

## 驗收條件

```bash
uv run pytest
uv run ruff check src tests benchmarks examples
uv run ruff format --check .
uv run mypy src
```

外加:
1. **三方對照表**,同一批題目、同一模型:
   | 模式 | 說明 |
   |---|---|
   | baseline | 單次呼叫 |
   | saccade（無工具） | M1 的狀態,作為對照 |
   | saccade（有工具） | 驗證環節真的運作 |
2. **收斂率 > 0** —— 若仍是 0/N,表示工具沒接上,不得宣稱完成
3. **每個數字附 McNemar p 值**,不顯著就寫不顯著
4. 消融實驗:拆開「多看幾次」與「工具驗證」各自的貢獻
5. 結果表進 README(**這是 M2 才做的事**)

---

## Task 1:圓偵測工具

**檔案**:`benchmarks/blindtest/tools/circles.py`(不放 `src/`,見下方註)

BlindTest 的 Touching Circles 只給圖片,但 `saccade.geometry.circles_overlap()` 需要圓心與半徑。中間缺的就是偵測。

```python
def detect_circles(image) -> list[tuple[Point, float]]:
    """Hough transform 找圓,回傳 [(圓心, 半徑), ...]"""
```

用 `cv2.HoughCircles`。參數要調,但**調參本身要用資料集的 metadata 驗證** —— 資料集有 `center_1`、`center_2`、`diameter`,可以直接算偵測誤差,不必靠肉眼。

包成 Tool:
```python
Tool(
    name="circle_geometry",
    fn=...,           # 偵測 → circles_overlap() → ToolResult
    ...
)
# ToolResult(value={"method": "circles_overlap", "overlap": bool, ...},
#            is_measurement=True)
```

**為什麼放 benchmarks/ 不放 src/**:規則 2。`cv2.HoughCircles` 是領域偵測器,引擎不得內建。這正是 `register_tool()` 存在的理由 —— benchmark 是引擎的「應用」之一。

**驗證**:
- 用 metadata 算偵測準確率,先確認偵測本身可信(偵測不準,驗證就是噪音)
- 偵測失敗時回傳什麼?(`is_measurement` 仍為 True 但值為 None?還是不回傳?)這決定 verifier 怎麼處理 —— **要明確決定並測試**

---

## Task 2:線段偵測工具

**檔案**:`benchmarks/blindtest/tools/lines.py`

同理,`count_line_intersections()` 需要線段座標。用 `cv2.HoughLinesP`。

Line Plot Intersections 的圖是藍紅兩條折線,可先用顏色分離再偵測,比純 Hough 穩。

**驗證**:同上,拿 metadata 驗偵測本身。

---

## Task 3:確認 verifier 真的在動

**這是 M2 最重要的一步,不是加功能,是驗證既有功能。**

註冊工具後重跑,檢查:
- `converged` 不再全是 False
- `confidence` 有分布,不是單一值
- 有 `verification.conflict` 非空的案例(工具真的推翻過 VLM)

**若這三項沒變,後面都不用做** —— 表示工具沒接上,或 verifier 的比對邏輯有問題。

M1 的 `_verifier.py` 用字串比對判斷衝突(`_boolean_conflict`、`_count_conflict`),粗糙且沒在真實資料上驗證過。這裡很可能會發現它不管用 —— **那就是 M2 的發現之一**,可能要改成讓 VLM 輸出結構化布林值。

---

## Task 4:三方對照

同一批題目(stratified,固定 seed)、同一模型,跑三種模式:

| 模式 | max_steps | tools |
|---|---|---|
| baseline | 1 | — |
| saccade-notools | 3 | — |
| saccade-tools | 3 | ✅ |

**樣本**:先 150 題確認方向,再決定要不要加大。**加大前先問成本**。

`baseline vs saccade-notools` 分離「多看幾次」的效果;`saccade-notools vs saccade-tools` 分離「工具驗證」的效果。這就是消融。

---

## Task 5:七任務全跑

M1 只跑 Touching Circles(7 選 1)。論文的 58.07% 是七任務平均,**單項數字不可與之比較**。

七個任務(dataset 實際名稱):
```
Touching Circles          Line Plot Intersections
Circled Letter            Olympic Counting - Circles
Olympic Counting - Pentagons
Nested Squares            Counting Grid - Blank Grids
Counting Grid - Word Grids    Subway Connections
```

**注意**:只有前兩個有現成的可計算裁判(M0 的 geometry)。其餘五個要嘛另外做工具,要嘛**誠實標示「無測量工具,僅多看幾次」** —— 不得假裝有驗證。

成本會不小,**跑之前先估算並確認**。

---

## Task 6:結果進 README

規則 8:每個數字附出處。表格要有:

- 模型名、樣本數、日期
- baseline / saccade-notools / saccade-tools 三欄
- **McNemar p 值**
- 收斂率
- 指向 `benchmarks/blindtest/results/` 的原始 JSON

**不顯著就寫不顯著。變差就寫變差。** spec 10.2 已寫明:消融說明「哪些策略有效/無效/為何」本身即為發現。

---

## 已知風險

| 風險 | 對策 |
|---|---|
| Hough 偵測不準,測量比 VLM 還爛 | 先用 metadata 驗偵測準確率。偵測不可信就不能當裁判 —— 這條是 spec 1.4 的底線 |
| verifier 字串比對在真實資料上不管用 | **已在 M1 發生過一次**(見上節)。改結構化輸出是備案:讓 VLM 輸出 `expect=` 的 Pydantic 模型,verifier 比對欄位而非猜字面。字串比對本質脆弱,M2 若再遇到就直接換掉 |
| 接上工具後仍不顯著 | 照實寫。M1 已經證明「小樣本假象」會誤導,誠實比好看重要 |
| 七任務成本過高 | 先兩個有裁判的任務,其餘視預算 |

---

## M2 不做的事

- 讓 VLM 自己規劃下一步(planner 仍規則式)
- 任何棒球相關(M3)
- 效能最佳化

---

## 成本紀律(M1 的教訓)

M1 期間我未經同意連續跑了四輪 API 呼叫。**M2 任何會花錢的跑批,執行前必須先報告預估用量並取得同意。**

已知單價參考(M1 實測):
- 150 題 baseline ≈ 72k tokens
- 150 題 saccade(3 steps) ≈ 262k tokens
