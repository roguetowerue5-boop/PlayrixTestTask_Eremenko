# Шрифт офферов

**Основной:** `default.ttf` = **Nunito** ExtraBold / variable (SIL OFL).

Нужен кириллический набор: прежний Fredoka Bold латиницу рисовал,
а русские буквы превращал в □□□. Nunito закрывает и RU, и EN, оставаясь
жирным казуальным гротеском.

## Где используется

| Место | Файл | Стиль |
|-------|------|--------|
| Название коллекции | ribbon / title | UPPERCASE, жёлтый градиент + коричневая обводка |
| Название пака | nameplate | UPPERCASE, белый + тёмно-синяя обводка |
| Футер `SET N/M` | footer | UPPERCASE |

## Файлы

| Файл | Назначение |
|------|------------|
| `default.ttf` | Дефолт для OfferBuilder (= Nunito variable) |
| `Nunito-Variable.ttf` | Исходник variable (wght 200–1000) |
| `Nunito-ExtraBold.ttf` | Копия variable (имя для совместимости) |
| `Fredoka-*.ttf` | Старый латинский набор (запасной) |
| `LilitaOne-Regular.ttf` | Альтернатива без кириллицы |
| `OFL-*.txt` | Лицензии |

Версия составляющей: `parts/font/playrix` (default в `parts/index.yaml`).
