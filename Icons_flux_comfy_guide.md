# Icons → Flux (ComfyUI) — Style Pack

Сводка после сравнения `Icons_style_guide.md` и разбора по `Icons/`.
Для ComfyUI: нода **CLIP TextEncode (Flux)** / Positive — вставляй блоки ниже как есть.

---

## 1. Сравнение с `Icons_style_guide.md`

| Тема | Гайд | Мой разбор | Вердикт для Flux |
|---|---|---|---|
| Два стиля | **A** = glossy 3D object, **B** = painterly character scene | Один «3D icon» + отдельно cat.5 персонажи | **Гайд правее**: персонажи (овца, рокер, пикник) действительно мягче/живописнее объектов. Для кино-наборов без золота почти всегда **Style A**. |
| Рендер объектов | glossy plastic/clay, bevels, specular | stylized 3D, toy-like, smooth | Совпадает |
| Композиция | 1 hero, ~60–70 % кадра, center, ¾ | single centered, icon silhouette | Совпадает |
| Фон Style A | radial **sunburst** / radial gradient + 1 hue | stripes, rays, waves, harlequin, spiral | **Гайд = дефолт**; полосы/волны/ромбы — варианты той же семьи «простой паттерн». Для теста Flux бери sunburst. |
| Тень Style A | soft drop/contact shadow, часто «floats» | cat.1 без тени (как в ТЗ) | Для **Icons/Flux** следуй **гайду** (мягкая тень). Запрет тени — только если явно просишь ТЗ cat.1. |
| Свет | soft key from **top**, fill, rim | upper-left key | Близко; для Flux: «soft key from above-left». |
| Цвет | hyper-saturated, hex-палитра с замеров | candy / saturated | **Гайд сильнее** — используй hex из гайда. |
| Формат промпта | **предложения** (Flux/Gemini) | keyword bank через запятую | Для Flux — **гайд**; keyword bank оставь для LoRA-captions / negative. |
| Negative | photo, muted, grit, text, flat 2D | + anime, pores, fur strands | Объединить оба |

**Итог:** для ComfyUI+Flux бери каркас из `Icons_style_guide.md` (Style A/B + предложения + hex). Мои правки: (1) sunburst не единственный паттерн, (2) для сдачи без золота не уходи в Style B, (3) negative расширь анти-anime/anti-photo.

---

## 2. Style Guide — что класть в workflow

### Positive — STYLE LOCK (всегда в начале Positive)

```
Playrix-like casual mobile game icon, glossy stylized 3D render,
single hero object centered in frame filling most of the image,
smooth rounded chunky shapes with soft beveled edges,
high-gloss plastic and clay materials with crisp specular highlights,
no surface noise, no grit, hyper-saturated candy colors,
bright soft studio lighting with key light from above-left,
gentle ambient fill, subtle rim light, soft contact shadow under the object,
clean polished match-3 collectible icon, readable silhouette,
high production value, no text, no logo, no watermark
```

### Positive — BACKGROUND (выбери один блок)

**A1 — классика Icons (sunburst):**
```
set against a single-hue radial sunburst background with soft vignette,
background color cyan #10BCE5 complementary to the subject
```

**A2 — мягкий radial gradient:**
```
set against a smooth radial gradient background in one saturated hue,
soft vignette, no clutter
```

**A3 — тихий паттерн (как часть Icons):**
```
set against a simple same-hue patterned backdrop
(soft diagonal stripes OR gentle waves OR quiet diamonds),
low-contrast pattern that never competes with the hero object
```

**A4 — лёгкая среда (не полная сцена):**
```
object rests on a minimal surface hint (wood plank or water rim),
backdrop stays a simple saturated gradient, not a full location
```

### Negative (Flux: если есть Negative CLIP — вставь; иначе допиши «Avoid: …» в конец Positive)

```
photorealistic, photograph, DSLR, cinematic still, film grain,
muted desaturated colors, dark moody atmosphere, harsh shadows,
gritty texture, rust, dirt, pores, individual hairs, fine fur strands,
mesh grille, tiny illegible details, cluttered background,
text, letters, numbers, logo, watermark, UI, frame, card border,
flat vector, flat 2D, anime, manga, illustration sketch, oil painting,
multiple competing objects, cropped subject
```

### Параметры ComfyUI (старт)

| Параметр | Старт | Зачем |
|---|---|---|
| Resolution | 1024×1024 (или 896×1024 ≈ 9:10) | иконка, квадрат/чуть выше |
| Steps | 28–32 | Flux sweet spot |
| Guidance / CFG | **3.0–4.0** | выше → жестче стиль, риск пластика; ниже → размытие стиля |
| Sampler | обычный для твоего Flux workflow | не принципиально |
| Seed | фиксируй при A/B | сравнение честное |

Цвета фона из замеров (подставляй в промпт одним):  
`#10BCE5` cyan · `#3B7BB8` sky · `#ECA411` gold · `#DA84A7` pink · `#8E4AD5` purple · `#40A848` green

Правило: **тёплый объект → холодный фон**, и наоборот.

---

## 3. Тестовые запросы (копипаст в Positive)

Собери Positive так: **STYLE LOCK + BACKGROUND + SUBJECT sentence**.

### TEST 1 — Style A, чистый объект (якорь / эталон sunburst)

```
Playrix-like casual mobile game icon, glossy stylized 3D render,
single hero object centered in frame filling most of the image,
smooth rounded chunky shapes with soft beveled edges,
high-gloss plastic and clay materials with crisp specular highlights,
no surface noise, hyper-saturated candy colors,
bright soft studio lighting with key light from above-left,
gentle ambient fill, soft contact shadow under the object,
clean polished match-3 collectible icon, no text, no logo.

A glossy stylized 3D render of a navy blue ship's anchor with a tan braided rope coiled around the shank, single hero object centered, thick rounded forms, soft matte-to-gloss transition on metal, set against a golden yellow #ECA411 radial sunburst background with soft vignette, complementary contrast, soft drop shadow, cheerful casual mobile-game icon.
```

### TEST 2 — Style A, кино-реквизит (то, что сдаёшь)

```
Playrix-like casual mobile game icon, glossy stylized 3D render,
single hero object centered in frame filling most of the image,
smooth rounded chunky shapes with soft beveled edges,
high-gloss plastic and clay materials with crisp specular highlights,
hyper-saturated candy colors, soft studio key from above-left,
soft contact shadow, clean match-3 icon, no text, no logo.

A glossy stylized 3D render of a vintage film camera with a short lens and a leather strap, single hero object slightly three-quarter view, polished black and silver body with bright specular hits, set against a deep purple #8E4AD5 radial sunburst background with soft vignette, hyper-saturated candy colors, soft studio lighting, gentle drop shadow, polished casual mobile-game collectible icon.
```

### TEST 3 — Style A, фон-паттерн (не только sunburst)

```
Playrix-like casual mobile game icon, glossy stylized 3D render,
single centered hero object, rounded chunky geometry, glossy toy materials,
soft above-left studio light, soft contact shadow, no text.

A glossy stylized 3D render of a purple folding hand fan with light wood ribs, open semi-circle, smooth candy-purple folds, set against a bright yellow backdrop with soft horizontal wave pattern in the same hue, floating product-shot framing, hyper-saturated colors, clean casual mobile-game icon.
```

### TEST 4 — только если нужны золотые/персонажи (Style B)

```
Playrix-like casual mobile game art, hand-painted 2.5D cartoon illustration,
soft airbrushed painterly shading, Pixar-lite expressive characters,
warm sunny lighting with gentle bloom, vibrant saturated colors,
wholesome lively scene, no text, no logo.

A hand-painted 2.5D cartoon illustration of a cheerful fluffy white sheep wearing a yellow scout hat and red neckerchief, aiming a wooden bow in a sunny stylized forest clearing, expressive friendly face, soft painterly shading, background trees with soft bokeh, warm daylight, high production casual mobile-game character card.
```

**Для сдачи кино-наборов без золота: гоняй TEST 1–3. TEST 4 не смешивай в один LoRA с объектами без нужды.**

---

## 4. Мини-чеклист «похоже на Icons?»

1. Объект один и крупный?  
2. Цвета «конфетные», не грязные?  
3. Блики есть, но нет фото-пор/зерна?  
4. Фон — один насыщенный цвет + sunburst/простой паттерн (не фотолокация), если это Style A?  
5. Нет текста / рамки карточки / UI?  
6. Силуэт читается с телефона?

Если 4–5 «нет» — понизь CFG до 3.0 и усиль STYLE LOCK; если слишком пластилин — CFG 3.5–4 и чуть конкретнее материал предмета.
