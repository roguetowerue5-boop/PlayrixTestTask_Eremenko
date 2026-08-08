"""Типизированные схемы всего пайплайна.

Каждый этап принимает и возвращает объект отсюда — это то, что делает
пайплайн проверяемым: между этапами нет свободных dict'ов.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Провайдеры и модели
# ---------------------------------------------------------------------------

class Capability(str, Enum):
    TEXT = "text"
    VISION = "vision"
    IMAGE = "image"


class ModelParam(BaseModel):
    """Одна запись в ротации варианта.

    Список таких записей = цепочка фолбэка: первая рабочая выигрывает.
    Формат намеренно повторяет model_params из SkyrimNet-пресетов.
    """
    name: str
    provider_id: str = "openrouter"
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    # Управление рассуждениями (формат OpenRouter):
    #   {"effort": "none"}   — не рассуждать
    #   {"exclude": true}    — рассуждать, но не отдавать рассуждения в ответ
    #   {"max_tokens": 1024} — жёсткий бюджет на рассуждения
    # Для этапов, где нужен строгий JSON, рассуждения только съедают лимит:
    # модель уходит думать и обрывается на середине ответа.
    reasoning: dict[str, Any] | None = None
    # image-специфика
    size: str | None = None
    n_reference_images: int | None = None
    # Всё остальное, что понимает конкретный провайдер картинок:
    # background, output_format, quality, aspect_ratio и прочее.
    # Уходит в тело запроса как есть, поверх default_params варианта.
    params: dict[str, Any] = Field(default_factory=dict)


class VariantConfig(BaseModel):
    """Конфигурация одной задачи-роли (brief / concept / critic / render / ...)."""
    model_name: str
    model_params: list[ModelParam] = Field(default_factory=list)
    default_params: dict[str, Any] = Field(default_factory=dict)
    timeout: int = 120
    # Триггер-слово своей LoRA. Ставится первым словом промпта: дообученная
    # модель узнаёт стиль по нему, и в середине текста токен теряет вес.
    trigger: str = ""

    def rotation(self) -> list[ModelParam]:
        if self.model_params:
            return self.model_params
        return [ModelParam(name=self.model_name)]


class Preset(BaseModel):
    id: str
    name: str
    variants: dict[str, VariantConfig]


# ---------------------------------------------------------------------------
# Шаблон оффера: слои и слоты
# ---------------------------------------------------------------------------

# part     — элемент из библиотеки составляющих, версия задаётся при генерации
# template — вложенный шаблон: набор собирается из готовых карточек, а не
#            пересобирает их раскладку у себя. Иначе рамка растягивается под
#            ячейку, а звёзды и подписи приходится дублировать.
LayerKind = Literal[
    "image_gen", "template", "part", "asset", "text", "solid", "gradient"
]


class Box(BaseModel):
    x: int
    y: int
    w: int
    h: int


class TextStyle(BaseModel):
    font: str = "default"
    size: int = 42
    color: str = "#FFFFFF"
    stroke_color: str | None = "#2A1A05"
    stroke_width: int = 4
    align: Literal["left", "center", "right"] = "center"
    uppercase: bool = True
    max_lines: int = 2
    shadow_offset: tuple[int, int] | None = (0, 3)


class Layer(BaseModel):
    """Один элемент оффера.

    image_gen — единственный тип, который уходит в нейросеть.
    asset/text/solid/gradient собираются локально и потому детерминированы.
    """
    id: str
    kind: LayerKind
    box: Box
    z: int = 0

    # kind == "asset"
    asset: str | None = None

    # kind == "part" — ссылка на тип составляющей из библиотеки.
    # Конкретная версия приходит из запроса на генерацию, а не из шаблона:
    # шаблон описывает раскладку, версии — внешний вид.
    part: str | None = None
    part_asset: str | None = None     # какой файл версии брать, ${rarity} допустим

    # kind == "template" — id вложенного шаблона (например "card")
    template: str | None = None

    # kind == "image_gen"
    prompt_slot: str | None = None      # имя слота в контент-плане
    category: str | None = None         # правило композиции (1..5 из ТЗ)
    mask: str | None = None             # опциональная маска скругления/формы
    fit: Literal["cover", "contain", "stretch"] = "cover"

    # kind == "text"
    text_slot: str | None = None
    style: TextStyle = Field(default_factory=TextStyle)

    # kind == "solid" / "gradient"
    color: str | None = None
    color_to: str | None = None

    # общее
    opacity: float = 1.0
    rotate: float = 0.0
    repeat_for: str | None = None       # имя коллекции: слой размножается по сетке
    grid: tuple[int, int] | None = None  # (cols, rows) для repeat_for
    gap: int = 12


class OfferTemplate(BaseModel):
    """Декларативное описание оффера как набора слоёв."""
    id: str
    name: str
    canvas_w: int
    canvas_h: int
    background: str = "#00000000"
    layers: list[Layer]

    def gen_layers(self) -> list[Layer]:
        return [l for l in self.layers if l.kind == "image_gen"]

    def text_layers(self) -> list[Layer]:
        return [l for l in self.layers if l.kind == "text"]

    def art_layer(self) -> Layer | None:
        """Слой, отвечающий за генерацию арта.

        У набора это слой-шаблон: сами image_gen лежат внутри вложенной
        карточки, но количество элементов и имена слотов определяются здесь.
        """
        for layer in self.layers:
            if layer.kind in ("image_gen", "template"):
                return layer
        return None

    def element_count(self, default: int = 1) -> int:
        layer = self.art_layer()
        if layer and layer.grid:
            return layer.grid[0] * layer.grid[1]
        return default


# ---------------------------------------------------------------------------
# Бриф и контент-план
# ---------------------------------------------------------------------------

class Brief(BaseModel):
    """Результат разбора свободного заказа продюсера."""
    theme: str
    genre: str | None = None
    mood: list[str] = Field(default_factory=list)
    palette: list[str] = Field(default_factory=list)
    era: str | None = None
    must_include: list[str] = Field(default_factory=list)
    must_avoid: list[str] = Field(default_factory=list)
    notes: str = ""
    # True, если модель была недоступна и заказ взят как есть, без разбора.
    # Попадает в отчёт прогона: иначе деградация выглядит как норма.
    degraded: bool = False


class ElementSpec(BaseModel):
    """Наполнение одного image_gen-слота."""
    slot: str
    subject: str                       # что изображено
    category: str = "1"                # правило композиции
    prompt: str = ""                   # собирается PromptBuilder'ом
    seed: int | None = None
    rarity: int = 1


class OfferPlan(BaseModel):
    """Один вариант наполнения оффера. Fan-out делается на этом уровне."""
    variant_id: str
    title: str
    concept: str                       # чем этот вариант отличается от других
    elements: list[ElementSpec]
    texts: dict[str, str] = Field(default_factory=dict)
    palette: list[str] = Field(default_factory=list)
    # Какие версии составляющих использовать: {"frame": "classic", ...}.
    # Пусто — берутся версии по умолчанию из библиотеки.
    parts: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Генерация и QC
# ---------------------------------------------------------------------------

class GeneratedAsset(BaseModel):
    slot: str
    path: str
    model: str
    provider: str
    seed: int | None = None
    prompt: str = ""
    attempt: int = 1
    cost_usd: float = 0.0
    elapsed_s: float = 0.0


class QCVerdict(BaseModel):
    slot: str
    passed: bool
    scores: dict[str, bool] = Field(default_factory=dict)
    reason: str = ""
    fix_hint: str = ""


class OfferResult(BaseModel):
    plan: OfferPlan
    assets: list[GeneratedAsset] = Field(default_factory=list)
    verdicts: list[QCVerdict] = Field(default_factory=list)
    composite_path: str | None = None
    needs_review: list[str] = Field(default_factory=list)


class RunReport(BaseModel):
    run_id: str
    template_id: str
    preset_id: str
    brief: Brief
    offers: list[OfferResult] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    total_images: int = 0
    rejected_images: int = 0
    elapsed_s: float = 0.0
