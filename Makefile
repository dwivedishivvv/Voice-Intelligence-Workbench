.PHONY: up down clean logs test

up:      ; pwsh -File start.ps1
down:    ; pwsh -File start.ps1 -Stop -Down
clean:   ; docker compose down -v
logs:    ; docker compose logs -f
test:    ; python -m pytest tests/unit -q
