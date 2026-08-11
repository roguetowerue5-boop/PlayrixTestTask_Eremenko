"""Самопроверка без сети и без ключей.

    python tools/selftest.py

Прогоняет то, что ломается тише всего: подстановку в декларативных
провайдерах, фолбэк ротации, валидаторы правил набора и полную сборку
композита. Работает целиком на mock-провайдере и локальном HTTP-сервере,
поэтому годится и как smoke-тест после правок конфигов.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from app import config  # noqa: E402
from app.models import ElementSpec, ModelParam, OfferPlan, VariantConfig  # noqa: E402
from app.pipeline import stages  # noqa: E402
from app.pipeline.compose import Compositor  # noqa: E402
from app.pipeline.run import run_offer_generation  # noqa: E402
from app.prompts import engine  # noqa: E402
from app.prompts.engine import _DECORATORS as _PROMPT_DECORATORS  # noqa: E402
from app.providers.base import (  # noqa: E402
    ProviderError,
    decode_images,
    extract_path,
    substitute,
)
from app.providers.registry import Registry  # noqa: E402

PASSED, FAILED = 0, 0


def legacy_render(name: str, **ctx) -> str:
    """Рендер из prompts/_archive — для selftest старых пайплайнов."""
    try:
        return engine.render(name, **ctx)
    except FileNotFoundError:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined

        arch = config.PROMPTS_DIR / "_archive"
        env = Environment(
            loader=FileSystemLoader(str(arch)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        env.globals.update(_PROMPT_DECORATORS)
        return env.get_template(f"{name}.j2").render(**ctx)


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok    {name}")
    else:
        FAILED += 1
        print(f"  ПРОВАЛ {name}  {detail}")


# ---------------------------------------------------------------------------

def test_dependencies() -> None:
    """Каждый сторонний импорт приложения должен быть в requirements.txt.

    Забытая зависимость проявляется как «сервер не запускается» — падает
    импорт всего приложения, без намёка на причину. Ловится только так.
    """
    print("\n[0] Зависимости")
    import ast
    import re

    root = Path(__file__).resolve().parent.parent
    req_text = (root / "requirements.txt").read_text(encoding="utf-8")
    declared = {
        re.split(r"[<>=\[]", line.strip())[0].lower().replace("_", "-")
        for line in req_text.splitlines()
        if line.strip() and not line.startswith("#")
    }
    # Имя пакета и имя модуля совпадают не всегда.
    ALIASES = {
        "pil": "pillow", "yaml": "pyyaml", "jinja2": "jinja2",
        "dotenv": "python-dotenv", "multipart": "python-multipart",
        "python_multipart": "python-multipart",
    }
    STDLIB_OK = set(sys.stdlib_module_names) | {"app", "tools"}

    missing: list[str] = []
    for py in sorted((root / "app").rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in STDLIB_OK:
                    continue
                pkg = ALIASES.get(name.lower(), name.lower().replace("_", "-"))
                if pkg not in declared:
                    missing.append(f"{py.relative_to(root)}: {name} → {pkg}")

    check("все сторонние импорты объявлены в requirements.txt",
          not missing, "; ".join(sorted(set(missing))[:4]))
    check("python-multipart объявлен (нужен экрану «Составляющие»)",
          "python-multipart" in declared)


def test_config() -> None:
    print("\n[1] Конфигурация")
    presets = {p["id"] for p in config.list_presets()}
    check("пресеты найдены", bool(presets), str(presets))
    for pid in presets:
        p = config.load_preset(pid)
        check(f"пресет {pid}: есть default", "default" in p.variants)
        check(f"пресет {pid}: неизвестный вариант наследует default",
              config.resolve_variant(p, "нет-такого").model_name == p.variants["default"].model_name)
    tpls = {t["id"] for t in config.list_templates()}
    check("шаблоны офферов найдены", bool(tpls), str(tpls))
    check("style_bible загружается", bool(config.style_bible().get("categories")))


def test_prompts() -> None:
    print("\n[2] Промпты")
    names = engine.available_prompts()
    need = {"art", "brief", "collection_fill", "name_from_art",
            "suggest_description", "suggest_art_extra"}
    check("актуальные шаблоны на месте", need <= set(names), str(names))
    check("legacy-промпты убраны из активных",
          not ({"concepts", "critic", "segment", "part_extract", "dissect"} & set(names)),
          str(names))
    art = engine.render("art", subject="film reel", category="2", palette=["#f00"], extra="", seed=42, trigger="plrxcard")
    check("art.j2 Style A lock", "Playrix-like casual mobile game icon" in art)
    check("art.j2 subject sentence", "glossy stylized 3D render of film reel" in art)
    check("art.j2 hero lock", "HERO OBJECT (must match): film reel" in art)
    check("art.j2 category rule", "Cat 2" in art and "simple surface" in art)
    check("art.j2 без nothing else", "nothing else" not in art)
    check("art.j2 cat4 mountains allowed", "mountains" in engine.render(
        "art", subject="projector", category="4", palette=["#0af"], extra="", trigger="plrxcard").lower())
    fill = engine.render(
        "collection_fill",
        title="Test",
        description="farm tools",
        brief={"theme": "farm", "must_include": ["tractor"], "must_avoid": [],
               "mood": ["warm"], "palette": ["#0a0"], "genre": None},
        n_elements=10,
        n_variants=1,
        lang="en",
    )
    check("collection_fill видит бриф", "tractor" in fill)
    check("name_from_art рендерится", bool(engine.render("name_from_art")))
    check("парсер JSON терпит ```-обёртку",
          engine.parse_json_response('```json\n{"a": 1}\n```') == {"a": 1})
    check("парсер JSON терпит болтовню вокруг",
          engine.parse_json_response('Вот результат: [{"b": 2}] готово') == [{"b": 2}])


def test_substitution() -> None:
    print("\n[3] Декларативные провайдеры: подстановка и разбор ответа")
    out = substitute(
        {"m": "${model}", "res": {"w": "${width}"}, "seed": "${seed}", "mix": "a-${seed}-b"},
        {"model": "m1", "width": 1024, "seed": None},
    )
    check("тип значения сохраняется", isinstance(out["res"]["w"], int))
    check("None-ключ выкидывается", "seed" not in out)
    check("None внутри строки — пусто, не 'None'", out["mix"] == "a--b", out["mix"])

    resp = {"result": {"artifacts": [{"image_b64": "QQ=="}, {"image_b64": "Qg=="}]},
            "data": [{"b64_json": "Qw=="}], "images": ["RA=="]}
    for path, expect in (
        ("result.artifacts[*].image_b64", ["QQ==", "Qg=="]),
        ("data[*].b64_json", ["Qw=="]),
        ("images[*]", ["RA=="]),
        ("result.artifacts[1].image_b64", ["Qg=="]),
        ("нет.такого", []),
    ):
        check(f"путь {path}", extract_path(resp, path) == expect)
    check("битый base64 не роняет разбор",
          decode_images(["QQ==", "не-base64~~"], "base64") == [b"A"])


def test_rotation() -> None:
    print("\n[4] Ротация провайдеров и фолбэк")
    buf = BytesIO()
    Image.new("RGB", (8, 8), (200, 40, 40)).save(buf, format="PNG")
    png_b64 = base64.b64encode(buf.getvalue()).decode()
    seen: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: ANN002
            pass

        def do_POST(self):  # noqa: N802
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["auth"] = self.headers.get("X-Api-Key")
            if self.path == "/boom":
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"upstream down")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"result": {"artifacts": [{"image_b64": png_b64}]}}).encode()
            )

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["SELFTEST_KEY"] = "secret-123"

    spec = {
        "endpoint": "/v1/generate",
        "cost_per_image": 0.02,
        "body": {
            "model": "${model}", "text": "${prompt}",
            "resolution": {"width": "${width}", "height": "${height}"},
            "seed": "${seed}", "style_refs": "${reference_images}",
        },
        "response": {"images": "result.artifacts[*].image_b64", "encoding": "base64"},
    }
    cfg = {"providers": {
        "bad": {"kind": "custom_rest", "base_url": f"http://127.0.0.1:{port}",
                "capabilities": ["image"], "image": {**spec, "endpoint": "/boom"}},
        "good": {"kind": "custom_rest", "base_url": f"http://127.0.0.1:{port}",
                 "api_key_env": "SELFTEST_KEY", "auth_header": "X-Api-Key",
                 "auth_scheme": "", "capabilities": ["image"], "image": spec},
        "nokey": {"kind": "custom_rest", "base_url": "http://x",
                  "api_key_env": "SELFTEST_MISSING", "capabilities": ["image"], "image": spec},
    }}

    try:
        reg = Registry(cfg, offline=False)
        check("провайдер без ключа помечен ненастроенным",
              reg.available()["nokey"]["configured"] is False)

        variant = VariantConfig(model_name="m", model_params=[
            ModelParam(name="skipped", provider_id="nokey"),
            ModelParam(name="broken", provider_id="bad"),
            ModelParam(name="art-v2", provider_id="good", size="640x480", n_reference_images=1),
        ])
        res = asyncio.run(reg.image(variant, "vintage projector", seed=42,
                                    references=[b"a", b"b"]))
        check("фолбэк дошёл до рабочей модели", res.model == "art-v2", res.model)
        check("size из ModelParam применился",
              seen["body"]["resolution"] == {"width": 640, "height": 480})
        check("n_reference_images обрезал референсы", len(seen["body"]["style_refs"]) == 1)
        check("ключ ушёл в кастомный заголовок", seen["auth"] == "secret-123")
        check("картинка декодировалась",
              Image.open(BytesIO(res.images[0])).size == (8, 8))

        try:
            asyncio.run(reg.image(VariantConfig(
                model_name="m", model_params=[ModelParam(name="b", provider_id="bad")]), "x"))
            check("падение всей ротации даёт ProviderError", False)
        except ProviderError as e:
            check("падение всей ротации даёт ProviderError", "HTTP 500" in str(e))
    finally:
        srv.shutdown()


def test_validators() -> None:
    print("\n[5] Валидаторы правил набора")
    bad = OfferPlan(variant_id="v1", title="t", concept="c", elements=[
        ElementSpec(slot="art", subject="лампа", category="1"),
        ElementSpec(slot="art", subject="Лампа", category="1"),   # дубль
        ElementSpec(slot="art", subject="катушка", category="2"),
    ])
    issues = stages._validate_plan(bad, 3)
    check("ловит дубли объектов", any("дубли" in i for i in issues), str(issues))
    check("ловит нехватку категорий", any("категор" in i for i in issues), str(issues))

    fixed = stages._repair_plan(bad, 3)
    subjects = [e.subject.lower() for e in fixed.elements]
    check("починка убирает дубли", len(set(subjects)) == len(subjects))

    ok = OfferPlan(variant_id="v2", title="t", concept="c", elements=[
        ElementSpec(slot="art", subject=f"объект {i}", category=str(i % 4 + 1))
        for i in range(8)
    ])
    check("валидный план проходит без замечаний", stages._validate_plan(ok, 8) == [])

    a = OfferPlan(variant_id="a", title="", concept="", elements=[
        ElementSpec(slot="art", subject="лампа"), ElementSpec(slot="art", subject="катушка")])
    b = OfferPlan(variant_id="b", title="", concept="", elements=[
        ElementSpec(slot="art", subject="Лампа"), ElementSpec(slot="art", subject="билет")])
    check("пересечение между вариантами находится",
          stages.cross_variant_overlap([a, b]) == {"a↔b": ["лампа"]})


def test_diagnostics() -> None:
    """Ошибка провайдера должна доезжать до пользователя с причиной.
    Раньше все отказы схлопывались в «не удалось ни за одну попытку»."""
    print("\n[5b] Диагностика отказов")
    from app.providers.rest import OpenAICompatProvider, _is_json_mode_error
    from app.providers.base import TextRequest  # noqa: F811

    state = {"mode": "credits", "seen": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: ANN002
            pass

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["seen"].append(body)

            def send(code, payload):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

            cases = {
                "credits": (402, {"error": {"message": "Insufficient credits"}}),
                "nomodel": (404, {"error": {"message": "No endpoints found"}}),
                "badkey": (401, {"error": {"message": "User not found."}}),
                "in200": (200, {"error": {"message": "Provider returned error"}}),
            }
            if state["mode"] == "nojson":
                if "response_format" in body:
                    send(400, {"error": {"message": "response_format is not supported"}})
                else:
                    send(200, {"choices": [{"message": {"content": '[{"variant_id":"v1"}]'}}]})
                return
            send(*cases[state["mode"]])

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        p = OpenAICompatProvider("or", {"base_url": f"http://127.0.0.1:{port}",
                                        "capabilities": ["text"]})
        for mode, marker in (("credits", "кредит"), ("nomodel", "не найдена"),
                             ("badkey", "ключ не принят"), ("in200", "Provider returned")):
            state["mode"] = mode
            try:
                asyncio.run(p.generate_text("m", TextRequest(prompt="x", json_mode=True), 10))
                check(f"отказ {mode} даёт ошибку", False)
            except ProviderError as e:
                check(f"причина видна: {mode}", marker in str(e), str(e)[:90])

        state["mode"] = "nojson"
        state["seen"].clear()
        res = asyncio.run(p.generate_text("m", TextRequest(prompt="x", json_mode=True), 10))
        check("json-режим не поддержан — есть повтор без него", len(state["seen"]) == 2)
        check("во втором запросе response_format убран",
              "response_format" not in state["seen"][1])
        check("вместо параметра просим JSON словами",
              "валидный JSON" in state["seen"][1]["messages"][-1]["content"])
        check("ответ всё же получен", "variant_id" in res.text)

        check("не всякая 400 считается json-проблемой",
              not _is_json_mode_error(Exception("HTTP 400 — bad request")))
    finally:
        srv.shutdown()

    # Обрыв ответа и съеденный рассуждениями бюджет выглядят как «битый
    # JSON» и уводят диагностику в сторону. Обе ситуации должны называться
    # своими именами.
    state3 = {"mode": "length", "seen": []}

    class LimitHandler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: ANN002
            pass

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state3["seen"].append(body)

            def send(code, payload):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

            mode = state3["mode"]
            if mode == "length":
                send(200, {"choices": [{"finish_reason": "length",
                                        "message": {"content": "{"}}],
                           "usage": {"completion_tokens": 8000}})
            elif mode == "reasoning_ate":
                send(200, {"choices": [{"finish_reason": "stop",
                                        "message": {"content": "", "reasoning": "…"}}],
                           "usage": {"completion_tokens_details":
                                     {"reasoning_tokens": 7900}}})
            elif "reasoning" in body:
                send(400, {"error": {"message": "reasoning is not supported"}})
            else:
                send(200, {"choices": [{"finish_reason": "stop",
                                        "message": {"content": '[{"variant_id":"v1"}]'}}]})

    srv3 = HTTPServer(("127.0.0.1", 0), LimitHandler)
    port3 = srv3.server_address[1]
    threading.Thread(target=srv3.serve_forever, daemon=True).start()
    try:
        prov3 = OpenAICompatProvider("or", {"base_url": f"http://127.0.0.1:{port3}",
                                            "capabilities": ["text"]})
        for mode, marker, label in (
            ("length", "обрезан по лимиту", "обрыв по max_tokens назван прямо"),
            ("reasoning_ate", "на рассуждения", "съеденный рассуждениями бюджет назван прямо"),
        ):
            state3["mode"] = mode
            try:
                asyncio.run(prov3.generate_text("m", TextRequest(prompt="x"), 10))
                check(label, False)
            except ProviderError as e:
                check(label, marker in str(e), str(e)[:90])

        state3["mode"] = "no_param"
        state3["seen"].clear()
        res3 = asyncio.run(prov3.generate_text(
            "m", TextRequest(prompt="x", params={"reasoning": {"effort": "none"}}), 10))
        check("модель не приняла reasoning — есть повтор без него",
              len(state3["seen"]) == 2 and "reasoning" not in state3["seen"][1])
        check("после повтора ответ получен", "variant_id" in res3.text)
    finally:
        srv3.shutdown()

    # Статус ключа и подробности из metadata — то, без чего 403 остаётся
    # загадкой. Плюс проверка, что тест провайдера не врёт: раньше он
    # дёргал публичный /models и был зелёным при полностью закрытом доступе.
    state2 = {"mode": "policy"}

    class KeyHandler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: ANN002
            pass

        def _send(self, code, payload):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def do_GET(self):  # noqa: N802
            if self.path.endswith("/key"):
                if state2["mode"] == "badkey":
                    return self._send(401, {"error": {"message": "Invalid API key"}})
                return self._send(200, {"data": {"label": "prod",
                                                 "limit_remaining": 7.5}})
            self._send(200, {"data": [{"id": "m1",
                                       "architecture": {"output_modalities": ["text"]}}]})

        def do_POST(self):  # noqa: N802
            if state2["mode"] == "policy":
                return self._send(403, {"error": {
                    "message": "Access denied by security policy",
                    "metadata": {"provider_name": "DeepSeek",
                                 "reasons": ["data_policy"]}}})
            self._send(200, {"choices": [{"message": {"content": "pong"}}]})

    srv2 = HTTPServer(("127.0.0.1", 0), KeyHandler)
    port2 = srv2.server_address[1]
    threading.Thread(target=srv2.serve_forever, daemon=True).start()
    try:
        prov = OpenAICompatProvider("or", {
            "base_url": f"http://127.0.0.1:{port2}", "capabilities": ["text"],
            "endpoints": {"chat": "/chat/completions", "models": "/models", "key": "/key"},
        })
        status = asyncio.run(prov.key_status())
        check("статус ключа читается", status.get("limit_remaining") == 7.5, str(status))

        state2["mode"] = "badkey"
        check("битый ключ виден в статусе",
              "401" in (asyncio.run(prov.key_status()) or {}).get("error", ""))

        state2["mode"] = "policy"
        try:
            asyncio.run(prov.generate_text("m", TextRequest(prompt="x"), 10))
            check("403 поднимает ошибку", False)
        except ProviderError as e:
            check("в 403 видно, какой провайдер отказал", "DeepSeek" in str(e), str(e)[:90])
            check("в 403 видна причина отказа", "data_policy" in str(e))

        # /models публичный и отвечает даже при закрытом чате — поэтому
        # проверкой доступа он служить не может.
        from app.models import Capability as Cap
        check("публичный /models отвечает при закрытом чате",
              asyncio.run(prov.list_models(Cap.TEXT)) == ["m1"])
    finally:
        srv2.shutdown()

    from app.providers.registry import _summarize
    check("одинаковые ошибки ротации схлопываются",
          _summarize(["одно и то же", "одно и то же"]) == "одно и то же")
    check("разные ошибки ротации перечисляются",
          "\n" in _summarize(["первая", "вторая"]))


def test_collection_context() -> None:
    """Правила набора описаны в style_bible один раз — и промпты, и валидатор
    должны читать именно их, а не свои копии."""
    print("\n[6] Контекст коллекции: один источник правды")
    coll = config.style_bible().get("collection") or {}
    check("секция collection есть в style_bible", bool(coll))
    check("размер набора задан", coll.get("set_size") == 10, str(coll.get("set_size")))

    rendered = engine.render(
        "collection_fill",
        title="т",
        description="тест",
        brief={"theme": "т", "must_include": [], "must_avoid": [],
               "mood": [], "palette": [], "genre": None},
        n_elements=10,
    )
    first_rule = (coll.get("rules") or [""])[0]
    check("правила из конфига попали в промпт", first_rule[:30] in rendered)
    check("раскладка редкостей попала в промпт", "1,1,2,2,3,3,4,4,5,5" in rendered)
    check("brief.j2 знает структуру коллекции",
          "16 наборов" in engine.render("brief", theme="т", wishes=""))

    # Золотые карточки выключены — категория 5 должна отсекаться валидатором.
    gold_on = (coll.get("gold_cards") or {}).get("enabled")
    check("золотые карточки выключены по ТЗ", gold_on is False, str(gold_on))
    check("валидатор требует 4 категории", stages.required_categories() == ("1", "2", "3", "4"),
          str(stages.required_categories()))

    with_gold = OfferPlan(variant_id="g", title="t", concept="c", elements=[
        ElementSpec(slot="art", subject=f"объект {i}", category=str(i % 5 + 1))
        for i in range(10)
    ])
    issues = stages._validate_plan(with_gold, 10)
    check("категория 5 при выключенных золотых отклоняется",
          any("золот" in i for i in issues), str(issues))

    fixed = stages._repair_plan(with_gold, 10)
    check("починка убирает категорию 5",
          "5" not in {e.category for e in fixed.elements},
          str({e.category for e in fixed.elements}))
    check("после починки покрыты все 4 категории",
          {e.category for e in fixed.elements} == {"1", "2", "3", "4"})


def test_parts() -> None:
    """Составляющие оффера: разбор эталона, версии, детерминированная сборка."""
    print("\n[6b] Составляющие: разбор и версии")
    from app import parts as parts_lib
    from app.config import load_preset, providers_config
    from app.pipeline import dissect
    from app.providers.registry import Registry

    check("библиотека не пуста", bool(parts_lib.defaults()), str(parts_lib.defaults()))
    kinds = {k["kind"] for k in parts_lib.list_kinds()}
    check("все типы из фидбека на месте",
          {"stars", "frame", "button", "badge", "nameplate", "font"} <= kinds, str(kinds))

    # Синтетический эталон с узнаваемыми зонами.
    w, h = 512, 720
    card = Image.new("RGBA", (w, h), (30, 20, 60, 255))
    d = ImageDraw.Draw(card)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=30, outline=(255, 215, 90, 255), width=16)
    d.rectangle([128, 28, 384, 100], fill=(255, 200, 60, 255))
    d.rounded_rectangle([56, 590, 456, 676], radius=18, fill=(60, 30, 15, 255))
    d.ellipse([384, 22, 486, 124], fill=(210, 50, 60, 255))
    buf = BytesIO()
    card.save(buf, format="PNG")

    reg = Registry(providers_config(), offline=True)
    version = "selftest-эталон"
    result = asyncio.run(dissect.dissect_card(
        reg, load_preset("offline"), buf.getvalue(), version, make_default=False))

    saved = {s["kind"] for s in result["saved"]}
    check("разбор нарезал рамку, звёзды, плашку и бейдж",
          {"frame", "stars", "nameplate", "badge"} <= saved, str(saved))
    check("шрифт сохранён описанием, а не картинкой", "font" in saved)

    fr = parts_lib.load_version("frame", version)
    img = Image.open(fr.asset_path("frame.png"))
    alpha = img.getchannel("A")
    # Без прозрачного окна рамка легла бы поверх арта и закрыла его.
    check("в рамке выбито окно под арт",
          alpha.getpixel((img.width // 2, img.height // 2)) == 0)
    check("края рамки остались непрозрачными", alpha.getpixel((8, 8)) == 255)
    check("якорь элемента сохранён", fr.anchor is not None, str(fr.anchor))

    font_ver = parts_lib.load_version("font", version)
    check("стиль шрифта извлечён", font_ver and font_ver.font is not None)

    # Сборка одной раскладки разными версиями должна давать разный результат,
    # но каждый — воспроизводимый.
    from app.config import load_template
    from app.pipeline.compose import Compositor

    tpl = load_template("card")
    plan = OfferPlan(variant_id="v", title="ТЕСТ", concept="",
                     palette=["#3B1F5E", "#160C2C"], texts={"title": "ТЕСТ"},
                     elements=[ElementSpec(slot="art", subject="объект", rarity=3)])
    a1 = Compositor(tpl, part_versions={"frame": "classic"}).render(plan, art={})
    a2 = Compositor(tpl, part_versions={"frame": "classic"}).render(plan, art={})
    b = Compositor(tpl, part_versions={"frame": version}).render(plan, art={})
    check("одинаковые версии дают одинаковый результат", a1.tobytes() == a2.tobytes())
    check("другая версия рамки меняет картинку", a1.tobytes() != b.tobytes())

    check("шаблон карточки ссылается на составляющие, а не на файлы",
          any(l.kind == "part" for l in tpl.layers))
    check("генерации подлежит только арт",
          [l.id for l in tpl.gen_layers()] == ["art"],
          str([l.id for l in tpl.gen_layers()]))

    # Набор обязан собираться из готовых карточек. Пока он раскладывал арт
    # и рамку по отдельности, рамка растягивалась под ячейку, а звёзд с
    # подписями в наборе не было вовсе.
    set_tpl = load_template("card_set")
    nested = [l for l in set_tpl.layers if l.kind == "template"]
    check("набор собирается вложенным шаблоном", len(nested) == 1, str(nested))
    check("вложен именно шаблон карточки",
          nested and nested[0].template == "card")
    check("количество элементов берётся из сетки набора",
          set_tpl.element_count() == 10, str(set_tpl.element_count()))
    check("слой арта у набора найден через art_layer",
          set_tpl.art_layer() and set_tpl.art_layer().kind == "template")

    set_plan = OfferPlan(
        variant_id="s", title="НАБОР", concept="",
        palette=["#3B1F5E", "#160C2C"], texts={"set_title": "НАБОР"},
        elements=[ElementSpec(slot="art", subject=f"объект {i}",
                              rarity=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5][i])
                  for i in range(10)],
    )
    sheet = Compositor(set_tpl).render(set_plan, art={})
    check("композит набора нужного размера",
          sheet.size == (set_tpl.canvas_w, set_tpl.canvas_h), str(sheet.size))
    check("в наборе нарисовано больше одного оттенка",
          len(sheet.convert("RGB").getcolors(maxcolors=200000) or []) > 100)

    for kind in list(saved):
        try:
            parts_lib.delete_version(kind, version)
        except KeyError:
            pass


def test_segment_and_assemble() -> None:
    """Сегментация UI-скина и сборка экрана из готового арта."""
    print("\n[6c] Сегментация UI и сборка")
    from app import parts as parts_lib
    from app.config import available_skins, load_preset, load_template, providers_config
    from app.pipeline import segment
    from app.pipeline.compose import Compositor
    from app.providers.registry import Registry

    # Синтетический «скин»: панель с лентой, слотом и футером.
    w, h = 1200, 700
    skin = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(skin)
    d.rounded_rectangle([205, 12, 1000, 676], radius=40, fill=(240, 124, 30, 255))
    d.rounded_rectangle([205, 12, 1000, 118], radius=30, fill=(199, 74, 11, 255))
    d.rounded_rectangle([305, 195, 400, 383], radius=18, fill=(200, 215, 240, 255))
    d.rounded_rectangle([315, 205, 390, 340], radius=12, fill=(20, 12, 30, 255))
    d.rounded_rectangle([310, 342, 395, 375], radius=10, fill=(240, 124, 30, 255))
    d.rounded_rectangle([470, 616, 730, 662], radius=14, fill=(250, 220, 170, 255))
    buf = BytesIO()
    skin.save(buf, format="PNG")

    reg = Registry(providers_config(), offline=True)
    version = "selftest-скин"
    result = asyncio.run(segment.segment_skin(
        reg, load_preset("offline"), buf.getvalue(), version, make_default=False))

    saved = {s["kind"] for s in result["saved"]}
    check("сегментация нашла элементы экрана",
          {"panel", "ribbon", "frame", "nameplate"} <= saved, str(saved))
    check("сетка слотов определена",
          (result["slots"] or {}).get("cols") == 5, str(result["slots"]))
    check("референсы стиля сохранены", len(result["references"]) >= 3,
          str(len(result["references"])))
    check("скин виден в списке доступных", version in available_skins())

    # Референсы должны браться именно из выбранного скина — это и есть
    # «максимальная важность»: модель видит куски того самого интерфейса.
    from app.config import reference_images
    refs = reference_images(6, skin=version)
    check("референсы читаются по имени скина", len(refs) >= 3, str(len(refs)))
    check("у несуществующего скина берётся общая папка",
          isinstance(reference_images(6, skin="нет-такого"), list))

    # Затирание чужого текста на плашке.
    plate = Image.new("RGBA", (200, 60), (240, 124, 30, 255))
    ImageDraw.Draw(plate).rectangle([70, 20, 130, 40], fill=(255, 255, 255, 255))
    cleaned = segment.clean_text_area(plate)
    centre = cleaned.getpixel((100, 30))
    check("надпись на плашке затёрта", centre[:3] != (255, 255, 255), str(centre))

    # Экран собирается из элементов и не содержит ни одной генерации.
    screen = load_template("offer_screen")
    check("экран оффера состоит только из готовых элементов",
          not screen.gen_layers(), str([l.id for l in screen.gen_layers()]))
    check("арт в экране приходит вложенной карточкой",
          screen.art_layer() and screen.art_layer().kind == "template")
    # Крестик и футер в шаблоне не нужны: первый нарисован на ленте,
    # второй впечатан в панель. Отдельными слоями они дублировали то, что
    # и так приезжает вместе с носителем.
    screen_parts = {l.part for l in screen.layers if l.kind == "part"}
    check("в экране есть панель и лента", {"panel", "ribbon"} <= screen_parts)
    check("крестика и футера в шаблоне нет",
          not ({"close", "footer"} & screen_parts), str(screen_parts))

    plan = OfferPlan(
        variant_id="s", title="ЭКРАН", concept="",
        palette=["#F07C1E", "#C74A0B"], texts={"set_title": "ЭКРАН"},
        elements=[ElementSpec(slot="art", subject=f"объект {i}", rarity=i % 5 + 1)
                  for i in range(10)],
    )
    img = Compositor(screen).render(plan, art={})
    check("композит экрана нужного размера",
          img.size == (screen.canvas_w, screen.canvas_h), str(img.size))

    for kind in list(saved):
        try:
            parts_lib.delete_version(kind, version)
        except KeyError:
            pass
    # segment_skin кладёт исходник в style/skins — без этой строки тестовый
    # скин оставался в списке и всплывал в интерфейсе.
    from app import skins as skins_lib

    skins_lib.delete_skin(version)
    refs_dir = segment.references_dir(version)
    if refs_dir.exists():
        import shutil

        shutil.rmtree(refs_dir, ignore_errors=True)


def test_regen() -> None:
    """Перерисовка UI-элементов по образцу и вырезание фона."""
    print("\n[6d] Регенерация элементов")
    from app import parts as parts_lib
    from app.config import load_preset, providers_config
    from app.pipeline import matting, regen
    from app.providers.registry import Registry

    # Вырезание фона: заливка от краёв не должна выедать такой же цвет
    # внутри самого элемента.
    art = Image.new("RGB", (300, 300), matting.CHROMA)
    d = ImageDraw.Draw(art)
    d.ellipse([70, 70, 230, 230], fill=(240, 124, 30))
    d.ellipse([130, 130, 170, 170], fill=matting.CHROMA)   # «дырка» цвета фона
    cut = matting.remove_background(art)
    check("фон срезан по краям", cut.getpixel((4, 4))[3] < 20)
    check("тело элемента цело", cut.getpixel((100, 150))[3] > 200)
    check("цвет фона внутри элемента не выеден",
          cut.getpixel((150, 150))[3] > 200)
    check("покрытие в разумных пределах", 0.05 < matting.coverage(cut) < 0.6,
          str(round(matting.coverage(cut), 3)))

    # Замкнутая область фона внутри рамки — её заливка от краёв не достаёт.
    frame = Image.new("RGB", (300, 300), matting.CHROMA)
    ImageDraw.Draw(frame).rounded_rectangle(
        [60, 40, 240, 260], radius=20, outline=(200, 215, 240), width=18)
    key = matting.detect_key(frame)
    step1 = matting.remove_background(frame, key=key, kill_spill=False)
    check("окно рамки заливкой от краёв НЕ вырезается",
          step1.getpixel((150, 150))[3] > 200)
    step2 = matting.remove_enclosed(step1, key=key)
    check("окно рамки вырезается поиском замкнутых областей",
          step2.getpixel((150, 150))[3] == 0)
    check("кант рамки при этом цел", step2.getpixel((150, 45))[3] > 200)

    # Полоска звёзд собирается кодом из одного атома — так шаг ровный,
    # а звёзды одинаковые.
    star = Image.new("RGBA", (60, 60), (0, 0, 0, 0))
    ImageDraw.Draw(star).ellipse([5, 5, 55, 55], fill=(255, 200, 60, 255))
    strip3 = regen.star_strip(star, 3)
    strip5 = regen.star_strip(star, 5)
    check("полоска из 3 звёзд шире одной", strip3.width > star.width)
    check("полоска из 5 шире, чем из 3", strip5.width > strip3.width)
    check("высота полоски не растёт", strip5.height == strip3.height)

    # Полный цикл на mock: образцы берутся из версии сегментации.
    src_version = "selftest-src"
    for kind, size in (("frame", (200, 280)), ("stars", (80, 80)),
                       ("nameplate", (240, 60))):
        sample = Image.new("RGBA", size, (240, 124, 30, 255))
        buf = BytesIO()
        sample.save(buf, format="PNG")
        parts_lib.save_version(
            kind, src_version,
            assets={(parts_lib.PART_KINDS[kind]["assets"] or ["a.png"])[0]: buf.getvalue()},
            source="selftest", make_default=False)

    reg = Registry(providers_config(), offline=True)
    out = asyncio.run(regen.regenerate_skin(
        reg, load_preset("offline"), src_version, "selftest-gen",
        make_default=False))

    made = {p["kind"] for p in out["parts"]}
    check("перерисованы все три типа", {"frame", "stars", "nameplate"} <= made,
          str(made) + " провалы: " + str([f["kind"] for f in out["failed"]]))

    stars_v = parts_lib.load_version("stars", "selftest-gen")
    check("звёзды разложены на пять вариантов",
          sorted(stars_v.assets) == [f"{n}.png" for n in range(1, 6)],
          str(stars_v.assets))

    frame_v = parts_lib.load_version("frame", "selftest-gen")
    fimg = Image.open(frame_v.asset_path("frame.png"))
    check("у перерисованной рамки прозрачное окно",
          fimg.getpixel((fimg.width // 2, fimg.height // 2))[3] == 0)
    check("в журнале есть покрытие и вердикт QC",
          any("coverage" in e and "qc" in e
              for p in out["parts"] for e in (p.get("journal") or [])))

    for kind in ("frame", "stars", "nameplate", "font"):
        for v in (src_version, "selftest-gen"):
            try:
                parts_lib.delete_version(kind, v)
            except KeyError:
                pass


def test_manual_markup() -> None:
    """Ручная разметка и полный экран как контекст регенерации."""
    print("\n[6e] Ручная разметка и контекст экрана")
    from app import parts as parts_lib, skins
    from app.config import load_preset, providers_config
    from app.pipeline import regen, segment
    from app.prompts import engine
    from app.providers.registry import Registry

    version = "selftest-manual"
    w, h = 1000, 600
    screen = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(screen)
    d.rounded_rectangle([100, 40, 900, 560], radius=30, fill=(240, 124, 30, 255))
    d.rounded_rectangle([100, 40, 900, 120], radius=24, fill=(199, 74, 11, 255))
    d.rounded_rectangle([200, 180, 320, 420], radius=16, fill=(200, 215, 240, 255))
    d.rounded_rectangle([212, 195, 308, 360], radius=10, fill=(20, 12, 30, 255))
    buf = BytesIO()
    screen.save(buf, format="PNG")

    # Исходник должен пережить перезагрузку: разметка и регенерация
    # происходят в разные заходы.
    skins.save_screen(version, buf.getvalue())
    check("исходный скин сохраняется", skins.has_screen(version))
    check("исходник читается обратно",
          skins.load_screen(version) == buf.getvalue())

    manual = {
        "parts": [
            {"kind": "panel", "box": {"x": 0.10, "y": 0.066, "w": 0.80, "h": 0.866}},
            {"kind": "ribbon", "box": {"x": 0.10, "y": 0.066, "w": 0.80, "h": 0.133}},
            {"kind": "frame", "box": {"x": 0.20, "y": 0.30, "w": 0.12, "h": 0.40}},
            {"kind": "art", "box": {"x": 0.212, "y": 0.325, "w": 0.096, "h": 0.275}},
        ],
        "summary": "размечено вручную",
    }
    skins.save_markup(version, manual, source="вручную")
    loaded = skins.load_markup(version)
    check("разметка сохраняется и читается", loaded and len(loaded["parts"]) == 4)
    check("источник разметки помечен", loaded.get("_source") == "вручную")
    check("скин виден в списке",
          any(s["version"] == version and s["has_markup"]
              for s in skins.list_skins()))

    # Готовая разметка должна отменять вызов vision.
    reg = Registry(providers_config(), offline=True)
    called: list[str] = []
    orig_analyze = segment.analyze

    async def spy(*a, **kw):
        called.append("analyze")
        return await orig_analyze(*a, **kw)

    segment.analyze = spy
    try:
        result = asyncio.run(segment.segment_skin(
            reg, load_preset("offline"), buf.getvalue(), version,
            markup=manual, make_default=False))
    finally:
        segment.analyze = orig_analyze

    check("с готовой разметкой vision не вызывается", not called, str(called))
    cut_kinds = {s["kind"] for s in result["saved"]}
    check("нарезано по ручным боксам", {"panel", "ribbon", "frame"} <= cut_kinds,
          str(cut_kinds))
    check("якорь элемента взят из разметки",
          parts_lib.load_version("frame", version).anchor is not None)

    # Полный экран и координаты области должны доезжать в промпт.
    prompt = legacy_render("part_regen", spec="A frame.", title="Рамка",
        region={"x": 0.2, "y": 0.3, "w": 0.12, "h": 0.4},
        chroma="chroma green", extra="")
    check("в промпте есть полный экран как референс 1",
          "REFERENCE IMAGE 1 is the full game screen" in prompt)
    check("в промпте есть координаты области", "x=0.2" in prompt)
    check("без экрана координаты не подставляются",
          "REFERENCE IMAGE 1" in legacy_render("part_regen", spec="x", title="t", region=None,
              chroma="c", extra="") and "x=0.2" not in legacy_render("part_regen", spec="x", title="t", region=None, chroma="c", extra=""))

    gen = asyncio.run(regen.regenerate_skin(
        reg, load_preset("offline"), version, f"{version}-gen",
        screen=buf.getvalue(), make_default=False))
    check("регенерация с контекстом экрана отработала",
          len(gen["parts"]) >= 3, str([p["kind"] for p in gen["parts"]]))

    for kind in cut_kinds | {"font"}:
        for v in (version, f"{version}-gen"):
            try:
                parts_lib.delete_version(kind, v)
            except KeyError:
                pass
    skins.delete_skin(version)
    refs = segment.references_dir(version)
    if refs.exists():
        import shutil

        shutil.rmtree(refs, ignore_errors=True)


def test_lasso() -> None:
    print("\n[6f] Вырезание по контуру (лассо)")
    import math

    from app.pipeline.segment import cut, cut_polygon

    # Круг в квадрате: внутри цело, за контуром пусто.
    solid = Image.new("RGBA", (200, 200), (255, 0, 0, 255))
    circle = [[0.5 + 0.45 * math.cos(t / 40 * 2 * math.pi),
               0.5 + 0.45 * math.sin(t / 40 * 2 * math.pi)] for t in range(40)]
    holed = cut_polygon(solid, circle, feather=0)
    check("внутри контура пиксели целы", holed.getpixel((100, 100))[3] == 255)
    check("за контуром прозрачно", holed.getpixel((2, 2))[3] == 0)
    check("две точки — не контур, кадр не трогаем",
          cut_polygon(solid, [[0.1, 0.1], [0.9, 0.9]], feather=0)
          .getpixel((2, 2))[3] == 255)
    check("точки за кадром обрезаются по границе",
          cut_polygon(solid, [[-5.0, -5.0], [9.0, -5.0], [9.0, 9.0], [-5.0, 9.0]],
                      feather=0).getpixel((100, 100))[3] == 255)

    # Треугольник внутри бокса: нижний правый угол обязан уйти в прозрачность.
    src = Image.new("RGBA", (400, 400), (0, 0, 0, 255))
    ImageDraw.Draw(src).rectangle([100, 100, 299, 299], fill=(0, 200, 255, 255))
    buf = BytesIO()
    src.save(buf, format="PNG")
    box = {"x": .25, "y": .25, "w": .5, "h": .5}
    tri = [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75]]

    by_poly = Image.open(BytesIO(
        cut(buf.getvalue(),
            {"parts": [{"kind": "badge", "box": box, "polygon": tri}]})
        ["badge"]["badge.png"]))
    by_box = Image.open(BytesIO(
        cut(buf.getvalue(), {"parts": [{"kind": "badge", "box": box}]})
        ["badge"]["badge.png"]))

    check("по контуру: угол вне треугольника пуст",
          by_poly.getpixel((by_poly.width - 6, by_poly.height - 6))[3] == 0)
    check("по контуру: угол внутри треугольника цел",
          by_poly.getpixel((6, 6))[3] > 200)
    check("по боксу тот же угол остаётся залитым",
          by_box.getpixel((by_box.width - 6, by_box.height - 6))[3] == 255)

    # Разметка с контуром должна пережить сериализацию в API-модель.
    from app.api_markup import MarkupPart

    part = MarkupPart(kind="badge", box=box, polygon=tri)
    check("контур переживает валидацию схемы",
          part.model_dump()["polygon"] == tri)
    check("без контура поле не появляется в разметке",
          "polygon" not in MarkupPart(kind="badge", box=box)
          .model_dump(exclude_none=True))


def test_image_request() -> None:
    """Форма запроса к Image API: тут молча ломалось всё остальное."""
    print("\n[3b] Запрос к Image API")
    import asyncio as aio
    import base64 as b64

    from app.config import providers_config
    from app.providers.base import ImageRequest
    from app.providers.rest import OpenRouterImageProvider

    seen: dict = {}

    class Spy(OpenRouterImageProvider):
        async def _post(self, url, payload, timeout):
            seen["url"], seen["payload"] = url, payload
            buf = BytesIO()
            Image.new("RGBA", (8, 8), (1, 2, 3, 255)).save(buf, format="PNG")
            return {"data": [{"b64_json": b64.b64encode(buf.getvalue()).decode()}]}

    cfg = providers_config()["providers"]["openrouter_images"]
    provider = Spy("openrouter_images", cfg)
    aio.run(provider.generate_image("google/gemini-3.1-flash-image", ImageRequest(
        prompt="p", width=200, height=100,
        reference_images=[b"screen-bytes", b"crop-bytes"],
        params={"background": "transparent"}), 30))

    pl = seen["payload"]
    refs = pl.get("input_references")
    # Раньше референсы уходили полем "image", которого у Image API нет.
    # OpenRouter молча выбрасывал его, модель рисовала с нуля по тексту —
    # снаружи это выглядело как «не держит стиль».
    check("референсы уходят в input_references", isinstance(refs, list) and len(refs) == 2,
          str(list(pl)))
    check("нет старого поля image", "image" not in pl)
    check("референс нужной формы",
          refs and refs[0].get("type") == "image_url"
          and refs[0]["image_url"]["url"].startswith("data:image/png;base64,"))
    check("параметры модели доезжают", pl.get("background") == "transparent")
    check("размер передан", pl.get("size") == "200x100")

    # Те же параметры, но пройденные через реестр: default_params варианта
    # применялись только к тексту, картинки их не видели.
    from app.config import load_preset
    from app.models import VariantConfig
    from app.providers.registry import Registry

    reg = Registry(providers_config(), offline=False)
    reg.providers["openrouter_images"] = provider
    # extract убран; проверяем merge default_params на роли render.
    base = load_preset("budget").variants["render"]
    raw = base.model_dump()
    raw["default_params"] = {"background": "transparent", "output_format": "png"}
    variant = VariantConfig(**raw)
    aio.run(reg.image(variant, "p", width=64, height=64, references=[b"x"]))
    check("default_params варианта доходят до картинки",
          seen["payload"].get("background") == "transparent",
          str(seen["payload"]))


def test_markup_quality() -> None:
    """Двухпроходная разметка и отсев заведомо неверных боксов."""
    print("\n[6j] Качество разметки")
    import asyncio as aio

    from app.config import load_preset, providers_config
    from app.pipeline import segment
    from app.prompts import engine
    from app.providers.registry import Registry
    from app.providers.rest import FALLBACK_SCREEN_MARKUP as REF

    check("эталонная разметка проходит валидацию",
          not segment.validate_markup(REF), str(segment.validate_markup(REF)))

    cell = segment.first_cell(REF["slots"])
    check("первая ячейка сетки вычисляется", cell is not None)
    check("ячейка уже всей сетки по ширине",
          cell and cell["w"] < float(REF["slots"]["box"]["w"]),
          str(cell))

    bad = {"parts": [
        {"kind": "close", "box": {"x": .1, "y": .1, "w": .5, "h": .5}},
        {"kind": "panel", "box": {"x": 0, "y": 0, "w": .02, "h": .02}},
        {"kind": "stars", "box": {"x": .9, "y": .9, "w": .3, "h": .3}},
        {"kind": "frame", "box": {"x": .2, "y": .2, "w": 0, "h": .1}},
    ], "slots": {}}
    said = segment.validate_markup(bad)
    check("крестик во весь экран пойман",
          any("close" in s and "велик" in s for s in said), str(said))
    check("панель размером с иконку поймана",
          any("panel" in s and "мал" in s for s in said))
    check("бокс за кадром пойман", any("за кадр" in s for s in said))
    check("нулевой размер пойман", any("нулевой" in s for s in said))
    check("пропавшая сетка замечена", any("сетка" in s for s in said))

    # Второй проход: координаты кропа обязаны стать координатами экрана.
    W, H = 1000, 600
    screen = Image.new("RGBA", (W, H), (20, 20, 30, 255))
    reg = Registry(providers_config(), offline=True)
    # Ответ даёт mock-провайдер: он читает подсказанную зону из промпта и
    # отвечает внутри неё. Подменять parse_json_response нельзя — один
    # фиксированный бокс не попадёт в зоны всех элементов сразу, и его
    # справедливо отбракует проверка «найденное лежит вне зоны».
    got = aio.run(segment.analyze_cell(
        reg, load_preset("offline"), screen, REF["slots"]))

    kinds = {p["kind"] for p in got}
    check("крупный план вернул элементы карточки",
          {"frame", "stars", "nameplate", "badge"} <= kinds, str(kinds))
    check("окно под арт достроено само", "art" in kinds, str(kinds))
    check("счётчик звёзд сохранён",
          any(p.get("stars_count") == 3 for p in got))
    check("точка на теле элемента доехала до разметки",
          any("at" in p for p in got), str(got[:1]))
    check("промпт крупного плана просит точку",
          "give `at`" in legacy_render("segment_cell", kinds={}))

    ref_frame = next(p["box"] for p in REF["parts"] if p["kind"] == "frame")
    new_frame = next(p["box"] for p in got if p["kind"] == "frame")
    check("рамка попала в ту же зону экрана, что эталонная",
          abs(new_frame["x"] - ref_frame["x"]) < 0.05
          and abs(new_frame["y"] - ref_frame["y"]) < 0.05,
          f"{new_frame} против {ref_frame}")
    check("все боксы крупного плана внутри кадра",
          all(0 <= p["box"]["x"] and 0 <= p["box"]["y"]
              and p["box"]["x"] + p["box"]["w"] <= 1.001
              and p["box"]["y"] + p["box"]["h"] <= 1.001 for p in got))

    check("без сетки крупный план не вызывается",
          aio.run(segment.analyze_cell(
              reg, load_preset("offline"), screen, {})) == [])

    # Список целей — конфиг, а не константа в коде: набор элементов у
    # каждого скина свой, и добавление ещё одного не должно требовать
    # правки Python.
    from app.config import save_segment_targets, segment_targets

    targets = segment_targets()
    check("цели читаются из конфига", len(targets) >= 5, str(len(targets)))
    check("у каждой цели есть зона и англоимя",
          all(len(t["zone"]) == 4 and t["en"] and t["level"] in ("screen", "cell")
              for t in targets))
    check("уровни разложены по обоим", {t["level"] for t in targets}
          == {"screen", "cell"})
    check("флаги counted и hollow доезжают из конфига",
          all(isinstance(t["counted"], bool) and isinstance(t["hollow"], bool)
              for t in targets))

    # Добавление своей цели должно доживать до чтения обратно.
    extra = dict(targets[0])
    extra.update({"kind": "selftest-target", "en": "test thing",
                  "hint": "A thing.", "level": "cell", "zone": [0.1, 0.2, 0.3, 0.4]})
    save_segment_targets(targets + [extra])
    try:
        again = segment_targets()
        added = next((t for t in again if t["kind"] == "selftest-target"), None)
        check("своя цель сохраняется и читается", added is not None)
        check("зона своей цели не поехала",
              added and added["zone"] == [0.1, 0.2, 0.3, 0.4], str(added))
        check("порядок целей сохраняется",
              [t["kind"] for t in again[:-1]] == [t["kind"] for t in targets])
    finally:
        save_segment_targets(targets)
    check("список вернулся к исходному",
          [t["kind"] for t in segment_targets()] == [t["kind"] for t in targets])

    # Зоны обязаны накрывать элементы там, где те реально лежат.
    cell_box = segment.first_cell(REF["slots"])
    ref_boxes = {p["kind"]: p["box"] for p in REF["parts"] if p.get("box")}
    for kind, zone in segment.CELL_ZONES.items():
        if kind not in ref_boxes:
            continue
        t = {"kind": kind, "zone": zone}
        b = ref_boxes[kind]
        ex = (b["x"] + b["w"] / 2 - cell_box["x"]) / cell_box["w"]
        ey = (b["y"] + b["h"] / 2 - cell_box["y"]) / cell_box["h"]
        zx, zy, zw, zh = t["zone"]
        check(f"зона накрывает {t['kind']}",
              zx <= ex <= zx + zw and zy <= ey <= zy + zh,
              f"центр ({ex:+.2f},{ey:+.2f}) зона ({zx:+.2f},{zy:+.2f})"
              f"..({zx + zw:+.2f},{zy + zh:+.2f})")

    # В модель уходит полный скин, зона — только подсказка в тексте. На
    # вырезанном куске модель не видит окружения и путает плашку карточки
    # с лентой заголовка: они похожи, а различить их можно лишь по месту.
    one = legacy_render("segment_one", title="quantity badge", hint="A tag.",
                        zone={"x0": 0.30, "y0": 0.25, "x1": 0.36, "y1": 0.31},
                        counted=False, repeated=True)
    check("промпт говорит про полный экран",
          "full game collection screen" in one)
    check("зона передана подсказкой", "x from 0.3 to 0.36" in one)
    check("координаты просят от всего кадра",
          "relative to the WHOLE image" in one)
    check("для повторяющегося есть оговорка",
          "repeats across the screen" in one)
    check("без зоны подсказки нет",
          "That is a hint" not in legacy_render("segment_one", title="x", hint="", zone=None,
              counted=False, repeated=False))

    # Найденное вне подсказанной зоны должно отбраковываться: увидев весь
    # экран, модель приносит похожий элемент из другого его конца.
    far = {"found": True, "box": {"x": 0.90, "y": 0.90, "w": 0.05, "h": 0.05},
           "at": [0.92, 0.92]}
    orig2 = engine.parse_json_response
    engine.parse_json_response = lambda text: far
    try:
        rejected = aio.run(segment._find_one(
            reg, load_preset("offline").variants["critic"],
            Image.new("RGBA", (1000, 600), (30, 60, 120, 255)),
            (200, 100, 400, 400), "stars",
            target={"kind": "stars", "en": "star", "hint": "",
                    "zone": [0.0, -0.2, 1.0, 0.3], "counted": False,
                    "level": "cell", "hollow": False, "title": "Звёзды"}))
    finally:
        engine.parse_json_response = orig2
    check("найденное вне зоны отбраковано", rejected is None, str(rejected))

    # Общий план не должен искать мелочь, крупный — крупное.
    check("уровни разметки не пересекаются",
          not (set(segment.SCREEN_LEVEL) & set(segment.CELL_LEVEL)))
    from app import parts as parts_lib

    big = legacy_render("segment", width=100, height=100, opaque=None,
                        kinds={k: parts_lib.PART_KINDS[k]
                               for k in segment.SCREEN_LEVEL})
    check("общий план просит сетку", "СЕТКУ КАРТОЧЕК" in big)
    check("общий план не просит звёзды", "stars —" not in big)


def test_verify() -> None:
    """Проверка нарезки сборкой обратно."""
    print("\n[6i] Сверка сборкой")
    import asyncio as aio
    from io import BytesIO as _B

    from app import parts as parts_lib, skins
    from app.config import load_preset, providers_config
    from app.pipeline import matting, segment, verify
    from app.providers.registry import Registry

    W, H = 800, 600
    screen = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(screen)
    d.rounded_rectangle([40, 30, 760, 570], radius=24, fill=(240, 124, 30, 255))
    d.rounded_rectangle([40, 30, 760, 110], radius=20, fill=(199, 74, 11, 255))
    # Один слот: рамка-кольцо, внутри плашка имени.
    d.rounded_rectangle([100, 160, 260, 420], radius=16,
                        outline=(200, 215, 240, 255), width=14)
    d.rounded_rectangle([115, 370, 245, 405], radius=8, fill=(255, 210, 74, 255))
    buf = BytesIO()
    screen.save(buf, format="PNG")

    markup = {
        "parts": [
            {"kind": "panel", "box": {"x": 0.05, "y": 0.05, "w": 0.90, "h": 0.90}},
            {"kind": "ribbon", "box": {"x": 0.05, "y": 0.05, "w": 0.90, "h": 0.133}},
            {"kind": "frame", "box": {"x": 0.125, "y": 0.267, "w": 0.20, "h": 0.433}},
            {"kind": "nameplate", "box": {"x": 0.144, "y": 0.617, "w": 0.162, "h": 0.058}},
        ],
        "slots": {"cols": 2, "rows": 1,
                  "box": {"x": 0.125, "y": 0.267, "w": 0.40, "h": 0.433}},
        "summary": "тест сверки",
    }

    pieces = segment.cut(buf.getvalue(), markup)
    original = Image.open(_B(buf.getvalue())).convert("RGBA")
    built, masks = verify.rebuild((W, H), markup, pieces)
    check("сборка того же размера", built.size == (W, H))
    check("в сборке что-то есть", verify.coverage_of(built) > 0.1
          if hasattr(verify, "coverage_of") else built.getbbox() is not None)

    report = verify.compare(original, built, markup, masks)
    check("отчёт по каждому размеченному типу",
          {r["kind"] for r in report["parts"]}
          == {"panel", "ribbon", "frame", "nameplate"},
          str([r["kind"] for r in report["parts"]]))

    # Рамка вырезана прямоугольником и потому содержит плашку — ровно тот
    # брак, ради которого сверка и делалась.
    swallowers = {s["kind"] for s in report["swallowed"]}
    check("поглощение плашки рамкой поймано", "frame" in swallowers,
          str(report["swallowed"]))
    check("подложку в поглощении не обвиняют",
          not ({"panel", "ribbon"} & swallowers), str(swallowers))

    # Смещения ячеек считаются от той, где элемент размечен.
    first = verify.card_offsets(markup, {"x": 0.125, "y": 0.267, "w": 0.2, "h": 0.4})
    second = verify.card_offsets(markup, {"x": 0.325, "y": 0.267, "w": 0.2, "h": 0.4})
    check("для первой ячейки смещения вперёд",
          first[0] == (0.0, 0.0) and first[1][0] > 0, str(first))
    check("для второй ячейки смещения назад",
          second[0][0] < 0 and second[1] == (0.0, 0.0), str(second))
    check("ячеек столько, сколько в сетке", len(first) == 2)

    # Дыра: элемент, который не вырезался, обязан попасть в брак.
    holed = {k: v for k, v in pieces.items() if k != "nameplate"}
    _, hmasks = verify.rebuild((W, H), markup, holed)
    hrep = verify.compare(original, verify.rebuild((W, H), markup, holed)[0],
                          markup, hmasks)
    check("пропавший элемент попадает в брак", "nameplate" in hrep["failed"],
          str(hrep["failed"]))

    check("карта расхождений строится",
          verify.diff_map(original, built).mode == "RGB")

    # Локальный режим не должен повторять попытку: он детерминирован.
    reg = Registry(providers_config(), offline=True)
    _, hist = aio.run(verify.verify_and_repair(
        reg, load_preset("offline"), screen=buf.getvalue(), markup=markup,
        pieces=pieces, raw_pieces=pieces, mode="local"))
    check("локальный режим проверяет один раз", len(hist) == 1, str(len(hist)))

    _, mhist = aio.run(verify.verify_and_repair(
        reg, load_preset("offline"), screen=buf.getvalue(), markup=markup,
        pieces=pieces, raw_pieces=pieces, mode="model", attempts=2))
    check("режим модели пробует повторно", len(mhist) >= 1, str(len(mhist)))
    check("история попыток пронумерована",
          [h["attempt"] for h in mhist] == list(range(1, len(mhist) + 1)))

    # Починка обязана чинить, а не только диагностировать.
    fixed, fhist = aio.run(verify.verify_and_repair(
        reg, load_preset("offline"), screen=buf.getvalue(), markup=markup,
        pieces=pieces, raw_pieces=pieces, mode="local"))
    last = fhist[-1]
    check("починка отработала и записана", bool(last.get("repaired")),
          str(last.get("repaired")))
    check("после починки поглощения нет",
          "frame" not in {s["kind"] for s in last["swallowed"]},
          str(last["swallowed"]))
    check("рамка после вычитания не пустая",
          matting.coverage(Image.open(_B(next(iter(fixed["frame"].values()))))
                           .convert("RGBA")) > 0.02)

    # Выеденный элемент откатывается к исходному вырезу.
    ghost = BytesIO()
    Image.new("RGBA", (200, 60), (0, 0, 0, 0)).save(ghost, format="PNG")
    broken = {**pieces, "ribbon": {"ribbon.png": ghost.getvalue()}}
    restored, rhist = aio.run(verify.verify_and_repair(
        reg, load_preset("offline"), screen=buf.getvalue(), markup=markup,
        pieces=broken, raw_pieces=pieces, mode="local"))
    check("пустой элемент возвращается к исходному вырезу",
          restored["ribbon"] == pieces["ribbon"],
          str(rhist[-1].get("repaired")))

    # Атомизированные судить по расхождению нельзя: одна звезда против
    # пяти в оригинале — это норма, а не брак.
    check("атомизированные исключены из оценки расхождения",
          "stars" in verify.ATOMIZED and "button" in verify.ATOMIZED)

    # Модель умеет вернуть красивую, но постороннюю картинку. Такое нельзя
    # класть в библиотеку: вызывающий откатится к вырезу из скрина.
    from app.pipeline import extract as extract_lib

    def poly(color, size=(200, 200)):
        im = Image.new("RGBA", size, (0, 0, 0, 0))
        ImageDraw.Draw(im).regular_polygon(
            (size[0] // 2, size[1] // 2, size[0] // 3), 5, fill=color)
        return im

    yellow, green = poly((255, 214, 9, 255)), poly((60, 220, 60, 255))
    check("тот же элемент проходит проверку",
          not extract_lib.looks_wrong(yellow, yellow))
    check("другой цвет отбраковывается",
          "цвет" in extract_lib.looks_wrong(green, yellow),
          extract_lib.looks_wrong(green, yellow))
    check("кадрирование не считается подменой",
          not extract_lib.looks_wrong(yellow.crop((20, 20, 180, 180)), yellow))
    check("другие пропорции отбраковываются",
          "пропорции" in extract_lib.looks_wrong(
              Image.new("RGBA", (400, 40), (255, 214, 9, 255)), yellow))
    check("серое без оттенка проверку не валит",
          not extract_lib.looks_wrong(
              Image.new("RGBA", (100, 100), (128, 128, 128, 255)),
              Image.new("RGBA", (100, 100), (120, 120, 120, 255))))

    # Автолассо: от точки внутри элемента до контура, без участия человека.
    scene = Image.new("RGBA", (300, 200), (30, 90, 170, 255))
    ImageDraw.Draw(scene).ellipse([90, 50, 210, 150], fill=(255, 214, 9, 255))
    box = (80, 40, 220, 160)
    traced = extract_lib.wand_cut(scene, [(0.5, 0.5)], box)
    check("автолассо обвело фигуру", traced is not None)
    if traced:
        cov = matting.coverage(traced)
        check("контур по фигуре, а не по боксу", 0.4 < cov < 0.95,
              f"{cov:.2f}")
        check("угол бокса остался прозрачным",
              traced.getpixel((2, 2))[3] < 60)

    # Дыра внутри — цифра на бейдже, надпись на плашке — должна зарасти.
    holed = Image.new("RGBA", (200, 200), (20, 80, 160, 255))
    dh = ImageDraw.Draw(holed)
    dh.ellipse([40, 40, 160, 160], fill=(250, 200, 20, 255))
    dh.ellipse([90, 90, 110, 110], fill=(90, 40, 10, 255))   # «цифра»
    filled = extract_lib.wand_cut(holed, [(0.32, 0.5)], (30, 30, 170, 170))
    check("дыра внутри элемента заращена",
          filled is not None and filled.getpixel(
              (filled.width // 2, filled.height // 2))[3] > 200)

    # Заливка, ушедшая на фон, должна отбраковываться по углам.
    check("заливка фона не проходит",
          extract_lib.wand_cut(scene, [(0.03, 0.03)], (0, 0, 300, 200)) is None)

    seeds = extract_lib.probe_seeds({"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4})
    check("пробные точки вокруг центра", len(seeds) == 7
          and abs(seeds[0][0] - 0.4) < 1e-6 and abs(seeds[0][1] - 0.4) < 1e-6)

    # Требование пустой середины должно попадать в промпт рамки.
    from app.prompts import engine

    hollow = legacy_render("part_extract", title="card slot frame",
                           chroma="green", region=None, has_screen=False,
                           atomize=False, hollow=True, extra="")
    check("для рамки просят пустую середину",
          "HOLLOW OUTLINE" in hollow and "must be a ring you can see through" in hollow)
    check("для остальных такого требования нет",
          "HOLLOW OUTLINE" not in legacy_render("part_extract", title="badge", chroma="green", region=None,
              has_screen=False, atomize=False, hollow=False, extra=""))
    check("рамка помечена как кольцо", "frame" in extract_lib.HOLLOW)

    # Умолчание должно быть предсказуемым, а не «как повезёт с моделью».
    html = (Path(__file__).resolve().parent.parent
            / "app" / "static" / "index.html").read_text(encoding="utf-8")
    # Умолчание — «Моделью»: на реальных скинах она отделяет элемент от
    # подложки там, где заливка по цвету не справляется.
    check("в интерфейсе по умолчанию модель",
          'mode: "model"' in html
          and '$(\'#mkModes [data-mode="model"]\').classList.add("pri")' in html)
    check("серверное умолчание тоже модель",
          'extract_mode: str = "model"' in (
              Path(__file__).resolve().parent.parent / "app" / "api_markup.py"
          ).read_text(encoding="utf-8"))

    for kind in ("panel", "ribbon", "frame", "nameplate", "font"):
        try:
            parts_lib.delete_version(kind, "selftest-verify")
        except KeyError:
            pass
    skins.delete_skin("selftest-verify")


def test_lora() -> None:
    """Своя LoRA: датасет, триггер-слово, подключение провайдера."""
    print("\n[6h] LoRA на стиль игры")
    import importlib.util

    from app.config import art_references, load_preset, preset_index, providers_config
    from app.models import ElementSpec
    from app.pipeline.stages import build_prompt

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "build_lora_dataset", root / "tools" / "build_lora_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Квадрат без искажения пропорций: растяжение LoRA выучила бы как стиль.
    tall = Image.new("RGB", (430, 480), (200, 40, 60))
    ImageDraw.Draw(tall).ellipse([100, 150, 330, 330], fill=(250, 220, 40))
    sq = mod.squarify(tall, 512)
    check("датасет: картинка становится квадратной", sq.size == (512, 512))
    check("датасет: пропорции объекта не поехали",
          abs(sq.width - sq.height) == 0
          and sq.getpixel((256, 256)) == (250, 220, 40))
    # Поля должны продолжать край, а не быть рамкой другого цвета: рамку
    # модель выучивает наравне со стилем.
    left_edge = sq.getpixel((2, 256))
    inner = sq.getpixel((sq.width // 2 - (512 - round(430 / 480 * 512)) // 2 - 4, 20))
    check("датасет: поля продолжают край, а не рамка",
          left_edge == (200, 40, 60), str(left_edge))

    wide = mod.squarify(Image.new("RGB", (900, 300), (10, 120, 90)), 256)
    check("датасет: широкая картинка тоже квадратится",
          wide.size == (256, 256) and wide.getpixel((128, 4)) == (10, 120, 90))

    check("датасет: триггер редкий, не общее слово",
          mod.TRIGGER == "plrxcard" and mod.TRIGGER.lower() not in
          ("playrix", "cartoon", "game", "card"))

    # Триггер обязан стоять первым словом промпта.
    el = ElementSpec(slot="art", subject="a watering can", rarity=3, category="1")
    with_trigger = build_prompt(el, ["#8E44AD"], trigger="plrxcard")
    check("триггер стоит первым словом", with_trigger.startswith("plrxcard, "))
    check("без триггера промпт прежний",
          build_prompt(el, ["#8E44AD"]).startswith("a watering can"))

    lora = load_preset("lora")
    check("пресет lora зарегистрирован",
          "lora" in [p["id"] for p in preset_index()["presets"]])
    check("у пресета lora задан триггер",
          lora.variants["render"].trigger == mod.TRIGGER,
          lora.variants["render"].trigger)
    rot = lora.variants["render"].rotation()
    check("своя LoRA идёт первой в ротации", rot[0].provider_id == "fal_lora")
    check("есть откат на обычные модели",
          any(m.provider_id == "openrouter_images" for m in rot[1:]),
          str([m.provider_id for m in rot]))

    prov = providers_config()["providers"]
    check("провайдер fal_lora описан", "fal_lora" in prov)
    check("fal_lora по умолчанию выключен",
          prov["fal_lora"].get("enabled") is False)
    check("fal_lora отдаёт ссылки, а не base64",
          prov["fal_lora"]["image"]["response"]["encoding"] == "url")

    # Ответ со ссылками должен докачиваться: раньше загрузчик знал только
    # base64 и такой ответ выглядел бы как «картинок нет».
    import asyncio as aio

    from app.providers.base import ImageRequest
    from app.providers.rest import CustomRESTProvider

    png = BytesIO()
    Image.new("RGB", (8, 8), (7, 7, 7)).save(png, format="PNG")

    class FakeFal(CustomRESTProvider):
        async def _post(self, url, payload, timeout):
            return {"images": [{"url": "https://cdn.example/a.png"}]}

        async def _fetch_images(self, urls, timeout):
            return [png.getvalue() for _ in urls]

    got = aio.run(FakeFal("fal_lora", prov["fal_lora"]).generate_image(
        "flux-lora", ImageRequest(prompt="p", width=64, height=64), 30))
    check("ответ со ссылками превращается в картинки",
          len(got.images) == 1 and got.images[0][:4] == b"\x89PNG")

    # Эталонные арты должны идти первыми среди референсов.
    refs = art_references(3)
    if refs:
        from app.config import reference_images
        mixed = reference_images(6, skin=None)
        check("эталонные арты идут первыми в референсах",
              mixed[:len(refs)] == refs)
    else:
        check("папка эталонных артов пуста — заполняется --art-refs", True)


def test_extract() -> None:
    print("\n[6g] Изъятие элемента из скина")
    import asyncio as aio

    from app import parts as parts_lib, skins
    from app.config import load_preset, providers_config
    from app.pipeline import extract, matting, segment
    from app.prompts import engine
    from app.providers.registry import Registry

    # Фигура на градиентном фоне: глобальный порог такое не берёт, потому
    # что «цвет фона» меняется от края к краю.
    w, h = 260, 180
    grad = Image.new("RGBA", (w, h))
    d = ImageDraw.Draw(grad)
    for y in range(h):
        t = y / (h - 1)
        d.line([(0, y), (w, y)],
               fill=(int(20 + 60 * t), int(60 + 40 * t), int(140 + 60 * t), 255))
    d.ellipse([90, 50, 170, 130], fill=(255, 214, 9, 255))

    matted = extract.auto_matte(grad)
    cov = matting.coverage(matted)
    check("градиентный фон снят", 0.05 < cov < 0.45, f"покрытие {cov:.2f}")
    check("тело элемента цело", matted.getpixel((130, 90))[3] > 200)
    check("угол стал прозрачным", matted.getpixel((3, 3))[3] < 40)

    # Элемент, доходящий до края: его цвет попадает в периметр, и наивный
    # отбор фона назначил бы фоном сам элемент.
    touch = Image.new("RGBA", (120, 120), (30, 70, 150, 255))
    ImageDraw.Draw(touch).rectangle([0, 40, 70, 80], fill=(255, 200, 10, 255))
    tm = extract.auto_matte(touch)
    check("элемент у края выделения не съеден",
          tm.getpixel((30, 60))[3] > 200)

    # Уже обведённое лассо трогать нельзя.
    lasso = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    ImageDraw.Draw(lasso).ellipse([20, 20, 80, 80], fill=(200, 60, 60, 255))
    kept, _ = extract.extract_local(lasso, lassoed=True)
    check("вырезанное лассо не перематируется",
          matting.coverage(kept) > 0.5, f"{matting.coverage(kept):.2f}")

    # Группа одинаковых фигур должна дать одну.
    row = Image.new("RGBA", (300, 80), (0, 0, 0, 0))
    dr = ImageDraw.Draw(row)
    for i in range(4):
        dr.ellipse([12 + i * 72, 16, 60 + i * 72, 64], fill=(255, 214, 9, 255))
    atoms = extract.split_atoms(row)
    check("группа разобрана на отдельные фигуры", len(atoms) == 4, str(len(atoms)))
    one = extract.pick_atom(atoms)
    check("из группы выбран один экземпляр",
          one is not None and one.width < row.width // 2,
          str(one.size if one else None))

    # Перемычка тоньше самих фигур должна рваться.
    bridged = Image.new("RGBA", (200, 90), (0, 0, 0, 0))
    db = ImageDraw.Draw(bridged)
    db.ellipse([10, 15, 80, 75], fill=(255, 214, 9, 255))
    db.ellipse([120, 15, 190, 75], fill=(255, 214, 9, 255))
    db.rectangle([80, 43, 120, 47], fill=(255, 214, 9, 255))
    check("тонкая перемычка разрывается",
          len(extract.split_atoms(bridged)) == 2,
          str(len(extract.split_atoms(bridged))))

    # Промпт обязан просить редактирование, а не рисование.
    p = legacy_render("part_extract", title="Звёзды",
                      chroma=matting.CHROMA_NAME, region={"x": .1, "y": .2, "w": .3, "h": .4},
                      has_screen=True, atomize=True, extra="")
    check("промпт заявлен как вырезание, не рисование",
          "background-removal edit" in p and "NOT an illustration task" in p)

    # Кириллица в англоязычном промпте читается моделью как надпись, которую
    # надо нарисовать: с русским «Название пака» она рисовала ровно эту
    # строку буквами. Поэтому у каждого типа есть английское имя.
    import re as _re

    for kind, meta in parts_lib.PART_KINDS.items():
        check(f"{kind}: есть английское имя", bool(meta.get("en")))
        check(f"{kind}: имя без кириллицы",
              not _re.search("[а-яА-ЯёЁ]", meta.get("en", "")), meta.get("en", ""))
    ru = legacy_render("part_extract", title=parts_lib.PART_KINDS["nameplate"]["en"],
                       chroma=matting.CHROMA_NAME, region=None, has_screen=False,
                       atomize=False, extra="")
    check("в промпте извлечения нет кириллицы", not _re.search("[а-яА-ЯёЁ]", ru))
    check("в промпте запрещено перерисовывать",
          "Copy them, do not repaint." in p)
    check("в промпте есть полный экран и кроп",
          "REFERENCE IMAGE 1 is the full game screen" in p
          and "REFERENCE IMAGE 2 is the crop" in p)
    check("в промпте есть координаты области", "x=0.1" in p)
    check("для группы просят оставить один экземпляр", "exactly ONE" in p)
    check("без группы такого требования нет",
          "exactly ONE" not in legacy_render("part_extract", title="t", chroma="c", region=None,
              has_screen=False, atomize=False, extra=""))

    # extract убран из активных пресетов UI; старый код берёт default.
    for pid in ("budget", "balanced", "premium", "offline"):
        pre = load_preset(pid)
        check(f"{pid}: нет роли extract (убрана из UI)",
              pre.variants.get("extract") is None)
        check(f"{pid}: нет роли copy (убрана из UI)",
              pre.variants.get("copy") is None)
        for role in ("default", "brief", "concept", "critic", "render"):
            check(f"{pid}: есть роль {role}", role in pre.variants)

    # Полный тракт на моке: три режима дают три разных результата.
    version = "selftest-extract"
    screen = Image.new("RGBA", (900, 500), (0, 0, 0, 0))
    ds = ImageDraw.Draw(screen)
    ds.rounded_rectangle([60, 40, 840, 460], radius=28, fill=(30, 95, 168, 255))
    ds.rounded_rectangle([160, 120, 320, 380], radius=18, fill=(200, 215, 240, 255))
    ds.ellipse([420, 150, 520, 250], fill=(255, 214, 9, 255))
    buf = BytesIO()
    screen.save(buf, format="PNG")

    manual = {"parts": [
        {"kind": "frame", "box": {"x": 0.177, "y": 0.24, "w": 0.178, "h": 0.52}},
        {"kind": "badge", "box": {"x": 0.466, "y": 0.30, "w": 0.111, "h": 0.20}},
    ], "summary": "тракт извлечения"}

    reg = Registry(providers_config(), offline=True)
    sizes: dict[str, tuple] = {}
    for mode in ("raw", "local", "model"):
        res = aio.run(segment.segment_skin(
            reg, load_preset("offline"), buf.getvalue(), f"{version}-{mode}",
            markup=manual, extract_mode=mode, make_default=False,
            save_references=False))
        check(f"режим {mode}: элементы сохранены",
              {"frame", "badge"} <= {s["kind"] for s in res["saved"]})
        meta = parts_lib.load_version("badge", f"{version}-{mode}")
        with Image.open(meta.asset_path(meta.assets[0])) as im:
            sizes[mode] = (im.size, matting.coverage(im.convert("RGBA")))

    check("сырой режим оставляет фон", sizes["raw"][1] > 0.95,
          f"{sizes['raw'][1]:.2f}")
    check("локальный режим фон снимает", sizes["local"][1] < 0.95,
          f"{sizes['local'][1]:.2f}")
    check("режим модели дал результат", sizes["model"][0][0] > 0)

    for mode in ("raw", "local", "model"):
        for kind in ("frame", "badge", "font"):
            try:
                parts_lib.delete_version(kind, f"{version}-{mode}")
            except KeyError:
                pass
        skins.delete_skin(f"{version}-{mode}")


def test_compositor() -> None:
    print("\n[7] Сборка оффера из элементов")
    tpl = config.load_template("card_set")
    plan = OfferPlan(
        variant_id="v1", title="ТЕСТ", concept="",
        palette=["#3B1F5E", "#160C2C"],
        texts={"set_title": "ТЕСТОВЫЙ НАБОР"},
        elements=[ElementSpec(slot="art", subject=f"объект {i}", rarity=i % 5 + 1)
                  for i in range(10)],
    )
    img = Compositor(tpl).render(plan, art={})
    check("композит нужного размера", img.size == (tpl.canvas_w, tpl.canvas_h), str(img.size))
    check("композит не пустой", img.convert("RGB").getbbox() is not None)
    # Считаем цвета на уменьшенной копии: у полноразмерного композита их
    # бывает больше лимита getcolors, и тот возвращает None. Сколько именно
    # цветов — зависит от того, какая версия рамки сейчас по умолчанию,
    # поэтому раньше тест падал не от поломки, а от смены дефолта.
    small = img.convert("RGB").resize((240, 180))
    check("плейсхолдеры нарисованы вместо отсутствующего арта",
          len(small.getcolors(maxcolors=240 * 180) or []) > 50)


def test_pipeline() -> None:
    print("\n[8] Полный прогон на mock-провайдере")
    report = asyncio.run(run_offer_generation(
        theme="Кино-коллекция, селфтест", template_id="card_set",
        preset_id="offline", n_variants=2, concurrency=8,
    ))
    check("вариантов столько, сколько просили", len(report.offers) == 2)
    for offer in report.offers:
        subjects = [e.subject.lower() for e in offer.plan.elements]
        check(f"{offer.plan.variant_id}: 10 элементов", len(offer.plan.elements) == 10)
        check(f"{offer.plan.variant_id}: объекты уникальны",
              len(set(subjects)) == len(subjects))
        check(f"{offer.plan.variant_id}: покрыты все 4 категории",
              {e.category for e in offer.plan.elements} >= set(stages.CATEGORIES))
        check(f"{offer.plan.variant_id}: композит записан",
              offer.composite_path and Path(offer.composite_path).exists())

    run_dir = config.RUNS_DIR / report.run_id
    for artefact in ("brief.json", "concepts.json", "contact_sheet.png",
                     "run_report.json", "run_report.md"):
        check(f"артефакт {artefact}", (run_dir / artefact).exists())
    check("отбраковка QC сохранена",
          any(p.name == "rejected" for p in run_dir.rglob("rejected")))
    print(f"        прогон {report.run_id}: {report.total_images} изображений, "
          f"{report.rejected_images} отбраковано, {report.elapsed_s} с")


def main() -> int:
    print("OfferForge · самопроверка (без сети и ключей)")
    for fn in (test_dependencies, test_config, test_prompts, test_substitution,
               test_image_request, test_rotation,
               test_validators, test_diagnostics, test_collection_context,
               test_parts, test_segment_and_assemble, test_regen, test_manual_markup,
               test_lasso, test_extract, test_markup_quality, test_verify,
               test_lora, test_compositor,
               test_pipeline):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            global FAILED
            FAILED += 1
            print(f"  ПРОВАЛ {fn.__name__} — исключение: {e!r}")

    print(f"\nИтог: пройдено {PASSED}, провалено {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
