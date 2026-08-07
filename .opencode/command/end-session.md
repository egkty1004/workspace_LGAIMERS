---
description: 세션 종료 루틴 — git 커밋 + Notion 작업 로그 기록 후 요약 보고
agent: build
---

세션을 마무리합니다. AGENTS.md에 정의된 **세션 종료 루틴**을 정확히 수행하세요:

1. **Git 커밋** — `cd ~/workspace_LGAIMERS && git add -A && git status`로 변경사항 확인 후, 있으면 `git commit -m "type: 요약"` 형식으로 커밋. 커밋할 내용이 없으면 스킵.
2. **Notion 작업 로그** — `notion_API-patch-block-children` 도구로 페이지 ID `3b55ed6b-28d5-818c-95ef-d0794d8500b6` (Aimers9 해커톤 작업로그) 마지막에 이번 세션 요약을 append. 형식:
   - `#### YYYY-MM-DD 세션` (heading_3)
   - **한 일** / **실험·결과** / **다음 할 일** (bulleted_list_item)
3. **종료 요약 보고** — 커밋 여부, Notion 기록 완료, 다음 세션 할 일을 사용자에게 짧게 보고.

사용자가 추가로 전달한 내용($ARGUMENTS)이 있으면 로그에 포함하세요.
