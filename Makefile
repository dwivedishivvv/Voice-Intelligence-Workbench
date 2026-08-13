.PHONY: up down logs test

up:      ; docker compose up -d --build
down:    ; docker compose down
clean:   ; docker compose down -v
logs:    ; docker compose logs -f api worker
test:    ; python -m pytest tests/unit -q
