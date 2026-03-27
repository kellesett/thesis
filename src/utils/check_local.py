"""
Проверяет какие модели доступны на локальном сервере (LOCAL_API_BASE).

Запуск:
    python src/utils/check_local.py
    python src/utils/check_local.py --ping        # только проверить доступность
    python src/utils/check_local.py --model foo   # проверить конкретную модель
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

from openai import OpenAI, APIConnectionError, APIStatusError


def make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key or "local")


def list_models(client: OpenAI) -> list[tuple[str, str]]:
    """Возвращает список (id, status) отсортированный по id."""
    response = client.models.list()
    return sorted(
        (m.id, getattr(m, "status", "—")) for m in response.data
    )


def ping(client: OpenAI, model: str) -> tuple[bool, str, float]:
    """Отправляет минимальный запрос чтобы убедиться что модель отвечает.
    Возвращает (success, reply, elapsed_sec)."""
    import time
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'ok'"}],
            max_tokens=5,
        )
        elapsed = time.perf_counter() - t0
        reply = resp.choices[0].message.content or ""
        return True, reply.strip(), elapsed
    except APIStatusError as e:
        return False, f"HTTP {e.status_code}: {e.message}", time.perf_counter() - t0
    except Exception as e:
        return False, str(e), time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверить модели на LOCAL_API_BASE")
    parser.add_argument("--base",  default=None, help="URL сервера (default: LOCAL_API_BASE из .env)")
    parser.add_argument("--key",   default=None, help="API ключ (default: LOCAL_API_KEY из .env)")
    parser.add_argument("--ping",  action="store_true", help="Пинговать каждую модель коротким запросом")
    parser.add_argument("--model", default=None, help="Проверить конкретную модель (implies --ping)")
    args = parser.parse_args()

    base_url = args.base or os.getenv("LOCAL_API_BASE")
    api_key  = args.key  or os.getenv("LOCAL_API_KEY", "local")

    if not base_url:
        print("[!!] LOCAL_API_BASE не задан. Укажи в .env или через --base")
        sys.exit(1)

    print(f"[>>] Сервер: {base_url}")
    client = make_client(base_url, api_key)

    # Список моделей
    try:
        models = list_models(client)
    except APIConnectionError:
        print(f"[!!] Не удалось подключиться к {base_url}")
        print("     Убедись что сервер запущен и URL правильный")
        sys.exit(1)
    except Exception as e:
        print(f"[!!] Ошибка при получении списка моделей: {e}")
        sys.exit(1)

    if not models:
        print("[??] Сервер отвечает, но список моделей пуст")
        sys.exit(0)

    print(f"\n[::] Доступные модели ({len(models)}):")
    for mid, status in models:
        print(f"      * {mid}  [{status}]")

    # Пинг
    to_ping: list[str] = []
    if args.model:
        to_ping = [args.model]
    elif args.ping:
        to_ping = [mid for mid, _ in models]

    if to_ping:
        print("\n[>>] Пингуем модели...")
        for m in to_ping:
            ok, reply, elapsed = ping(client, m)
            status = "[OK]" if ok else "[!!]"
            print(f"      {status}  {m}  >>  {reply!r}  ({elapsed:.2f}s)")

    print()


if __name__ == "__main__":
    main()
