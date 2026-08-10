# Playrix-Style Icon Set — Style Parameters for Gemini / Flux

Reference distilled from 159 icons in `C:\Playrix\Icons` (`CardC6_card_*.png`, 430×480 px).
Measured across the whole set: **mean saturation ≈ 67 %**, **mean brightness ≈ 78 %** → the look is *bright, vivid, high-contrast casual mobile-game art* (Gardenscapes / Township / Fishdom family).

The set splits into **two distinct styles**. Use Style A for single objects, Style B for character scenes.

---

## STYLE A — Glossy 3D Hero-Object Icon
*(the majority: trophy, soccer ball, calculator, anchor, camera, boxing gloves, ship's wheel, ice cream, etc.)*

**One-line style tag:**
`glossy stylized 3D render, single hero object, centered, radial sunburst background, hyper-saturated candy colors, soft studio lighting, mobile game icon`

| Parameter | Value |
|---|---|
| **Render** | Semi-realistic stylized 3D render; smooth rounded geometry, thick beveled edges, glossy plastic/clay surfaces, crisp specular highlights, no surface noise |
| **Subject** | ONE hero object, isolated, slightly oversized, fills ~60–70 % of frame |
| **Composition** | Centered, symmetrical, product-shot framing, slight 3/4 top-down angle, generous padding |
| **Background** | Radial sunburst rays **or** smooth radial gradient; ONE dominant saturated hue; soft vignette; sometimes a minimal hint of environment (wood floor, wall, water) |
| **Color** | Hyper-saturated, high vibrancy, candy/pop colors; strong complementary contrast between subject and background |
| **Lighting** | Bright even studio light, soft key from top, ambient fill, gentle rim light |
| **Shadow** | Soft contact/drop shadow directly beneath object; object often floats |
| **Finish** | Clean, polished, premium, readable at small size; app-icon quality |
| **Mood** | Cheerful, inviting, playful |

**Style keyword bank (Style A):**
`3D render · stylized 3D · glossy · high-gloss specular highlights · smooth rounded shapes · beveled edges · soft plastic material · clay render · single object · centered composition · isolated subject · radial sunburst background · sunburst rays · radial gradient · vignette · hyper-saturated · vivid candy colors · complementary color contrast · soft studio lighting · soft drop shadow · floating object · casual mobile game icon · match-3 icon · app icon · clean · polished · high detail`

---

## STYLE B — Painterly Cartoon Character Scene
*(the minority: pig / cow / sheep / chicken characters + chubby humans doing activities — picnic, waterslide, scooter, camping, jenga, lemonade stand, outdoor movie, etc.)*

**One-line style tag:**
`hand-painted 2.5D cartoon illustration, expressive anthropomorphic animal and stubby human characters, lively scene, warm sunny lighting, casual mobile game art`

| Parameter | Value |
|---|---|
| **Render** | 2.5D painterly cartoon illustration; hand-painted soft airbrushed shading; Pixar/DreamWorks-lite; semi-realistic cartoon |
| **Subject** | Expressive characters mid-action — anthropomorphic pig, cow, sheep, chicken, or stubby big-nosed humans; exaggerated friendly faces |
| **Composition** | Dynamic medium shot, characters inside an environment, clear storytelling moment, foreground/background depth |
| **Background** | Fuller scene (backyard, beach, poolside, campsite, street); soft atmospheric depth, background bokeh, gradient sky |
| **Color** | Warm and saturated, sunny daylight or golden-hour glow |
| **Lighting** | Soft cinematic sunlight, gentle bloom, warm highlights, soft shadows |
| **Finish** | Rich, painterly, storybook, high production value |
| **Mood** | Playful, comedic, wholesome, lively |

**Style keyword bank (Style B):**
`2.5D cartoon illustration · hand-painted · painterly · soft airbrushed shading · Pixar-style · DreamWorks-style · semi-realistic cartoon · anthropomorphic animal characters · expressive exaggerated faces · stubby cartoon humans · dynamic scene · storytelling composition · depth of field · background bokeh · warm sunny lighting · golden hour · soft bloom · vibrant · wholesome · casual mobile game art · storybook illustration`

---

## Shared DNA (both styles)
`Playrix style · casual mobile game art · Gardenscapes / Township / Fishdom aesthetic · vibrant saturated palette · clean and readable · friendly and approachable · high production value · rounded forms · soft shadows · no text, no watermark`

---

## Color Palette (measured from the set)

Dominant background hues actually sampled from the icons:

| Swatch | Hex | RGB | Use |
|---|---|---|---|
| Cyan / teal | `#10BCE5` | 16,188,229 | most common sunburst bg |
| Sky blue | `#3B7BB8` | 59,123,184 | cool bg |
| Light azure | `#58B4DA` | 88,180,218 | soft bg |
| Grass green | `#40A848` | 64,168,72 | nature bg |
| Golden yellow | `#ECA411` | 236,164,17 | warm sunburst |
| Amber / sand | `#DA9F4F` | 218,159,79 | beach / warm |
| Pink / magenta | `#DA84A7` | 218,132,167 | playful bg |
| Purple | `#8E4AD5` | 142,74,213 | premium / night bg |
| Deep red | `#8C2A3C` | 140,42,60 | dramatic bg |

Rule of thumb: **one saturated background hue, complementary to the object.** Warm object → cool bg, cool object → warm bg.

---

## Prompt Templates (natural language — best for Flux & Gemini)

Flux and Gemini/Imagen respond to **descriptive sentences**, not comma-salad. Fill the `[SUBJECT]` / `[COLOR]` slots.

**Style A template:**
> A glossy stylized 3D render of **[SUBJECT]**, single hero object centered in frame, semi-realistic rounded shapes with high-gloss highlights and soft beveled edges. Set against a **[COLOR]** radial sunburst background with a soft vignette. Hyper-saturated candy colors, bright soft studio lighting, gentle soft drop shadow beneath the object. Clean polished casual mobile-game icon, high detail, no text.

**Style B template:**
> A hand-painted 2.5D cartoon illustration of **[CHARACTER(S)]** **[DOING ACTIVITY]** in **[SETTING]**. Expressive exaggerated cartoon faces, soft airbrushed painterly shading, Pixar-lite style. Warm sunny lighting with gentle bloom, background depth with soft bokeh, vibrant saturated colors. Lively wholesome casual mobile-game art, high production value, no text.

**Add-ons that reinforce the look:**
`--ar 1:1` or `--ar 9:10` · `square icon crop` · `centered` · `vibrant` · (Flux) push guidance ~3–4 for painterly softness · (Gemini) end with "clean, high quality, game asset."

**Avoid (steers away from the look):**
`photorealistic · realistic photography · muted / desaturated colors · dark, gritty, moody · harsh shadows · cluttered background · text, logo, watermark · flat vector / flat 2D (unless you want that)`

---

## Worked examples

**A — object:**
> A glossy stylized 3D render of a golden trophy cup, single hero object centered, rounded polished metal with bright specular highlights and soft bevels, against a deep purple radial sunburst background with a soft vignette, hyper-saturated colors, soft studio lighting, gentle drop shadow, clean casual mobile-game icon, no text.

**A — object:**
> A glossy stylized 3D render of a red-and-white ship's lifebuoy floating on stylized water, centered, smooth rounded forms with glossy highlights, cyan `#10BCE5` sunburst background, vivid saturated palette, soft top light, soft shadow, polished match-3 game icon, no text.

**B — character scene:**
> A hand-painted 2.5D cartoon illustration of a cheerful cartoon pig and a cow enjoying a backyard picnic with burgers, expressive exaggerated happy faces, soft painterly airbrushed shading, warm golden-hour sunlight with gentle bloom, lush green background with soft bokeh, vibrant wholesome casual mobile-game art, no text.

---

## Subject / theme bank (what the set actually covers)

- **Sport:** soccer ball, basketball hoop, baseball glove, tennis racket, golf, boxing gloves, trophy, medal, sneakers, water bottle, whistle, water-polo goal
- **Park / garden:** rose arch, stone lion statue, street lamp, fountain, topiary, flower bed, bird feeder, garden swing, wrought-iron gate, chess table, watering can, apple basket, lawn mower
- **Beach / summer:** beach ball, umbrella, flamingo float, sunscreen, sunglasses, flip-flops, sandcastle, cooler, palm tree, waterslide, kiddie pool, ice cream, popsicle, iced drink, beach bag
- **Sea / nautical:** anchor, ship's wheel, lighthouse, barrel, captain, rowboat, telescope, buoy, treasure map, sailing ship, seahorse, jellyfish, submarine, message in a bottle, diving mask, flippers
- **Camping / scouts:** tent, campfire pot, bugle, binoculars, neckerchief, sleeping bag, camp stool, scout jacket, folding chair, signpost
- **Travel:** taxi, airplane, ticket, neck pillow, phone booth, camping trailer, suitcase, road trip, gas station, double-decker bus, tour binoculars, bike rack
- **Music / party:** electric guitar, bass guitar, drum kit, DJ deck, keyboard, stage light, microphone podium, balloons, wristband, concert crowd
- **Home / everyday:** retro computer, clock, calculator, books, diploma, vending machine, air conditioner, fan, coffee cup, ring box, camera, projector, locker
- **Amusement park:** ferris wheel, roller coaster, carousel, bumper car, bouncy castle, claw machine, popcorn cart, dart target
- **Characters (Style B):** pig, cow, sheep, chicken, stubby bearded man — picnic, scooter ride, waterslide, jenga, twister, lemonade stand, hammock, shopping, outdoor movie, camping
