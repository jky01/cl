# Project Rules

- 專案總目標：建構一條讓小模型能夠永續學習並逐步成為大模型的路線；新知識要能融入既有權重，避免災難性遺忘，且不能依賴 joint full retraining / 聯合全量學習作為主要解法。
- 每次 `qa/codex` 回覆都要朝這個目標推進：明確評估方案是否促進 knowledge-into-weights、continual retention、growth necessity / compute advantage、no-memory inference，以及是否避免全量聯合訓練。
- 啟動後每一分鐘監聽是否有未讀取的 `qa/claude` 最新文件。
- 未讀判斷方式：比較 `qa/claude/yyyy-mm-dd.hh.mm.ss.md` 與 `qa/codex/yyyy-mm-dd.hh.mm.ss.md` 的最新檔名時間；若 `qa/claude` 最新檔名時間晚於 `qa/codex` 最新檔名時間，代表還有新文件需要讀取。
- 如果有未讀取的新文件，讀取 `qa/claude` 目錄下最新文件，詳細分析，然後輸出 Markdown 到 `qa/codex/yyyy-mm-dd.hh.mm.ss.md`。
- 每次輸出的 `qa/codex` Markdown 內必須記錄此次讀取的 `qa/claude` 檔案名稱，方便追蹤已處理來源。
- 本機自動監聽由 `scripts/qa_claude_watch.sh` 執行，crontab 每分鐘呼叫一次；腳本使用 lock 避免重入。
- watcher 啟動 Codex 時必須使用 `gpt-5.5` 與 `model_reasoning_effort="xhigh"`；目前由腳本中的 `CODEX_MODEL` / `CODEX_REASONING_EFFORT` 預設值顯式指定。
