# Своя LoRA на стиль карточек Icons

## Быстрый старт в UI

1. Перезапусти `run.bat`
2. **AI → LoRA Trainer**
3. Вставь **FAL_KEY** → Сохранить
4. Датасет уже собран (159 Icons) — при необходимости «Собрать / обновить»
5. **Запустить обучение** (steps 1100 по умолчанию; 500 обычно мало для стиля)
6. Дождись COMPLETED → **Включить fal_lora**
7. В наполнении пресет **LoRA (Icons)**

OpenRouter ключ не трогай: текст и откат картинок остаются на нём.

## Файлы

```
lora/dataset/     159 пар PNG+TXT, триггер plrxcard
lora/dataset.zip  архив для fal (~127 МБ)
lora/jobs.json    журнал обучений (создаётся UI)
style/art-refs/   6 эталонов для OpenRouter без LoRA
```

## API

- `POST /api/lora/key` `{api_key}`
- `POST /api/lora/rebuild-dataset`
- `POST /api/lora/train` `{steps, trigger_word}`
- `GET  /api/lora/job`
- `POST /api/lora/apply` `{lora_url, enable:true}`
