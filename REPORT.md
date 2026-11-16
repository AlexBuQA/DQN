# Отчёт по домашней работе № 2

**Автор:** Александра Бужор  
**Тема:** DQN — от CartPole к LunarLander

## 1. Цель и состав решения

Реализованы Vanilla DQN, Double DQN и Dueling DQN. На `CartPole-v1` независимо обучаются обычный DQN и Dueling Double DQN. На дискретный `LunarLander-v3` переносится Dueling Double DQN с Huber loss: используются те же классы агента, replay buffer, функция взаимодействия и Double TD-target, меняются только размеры сети и гиперпараметры.

Сдаваемый артефакт — `MEPHI_RL26_hw2_DQN_Бужор_Александра.ipynb`. Модульная реализация находится в пакете `dqn/`, а конфигурации — в `configs/`.

## 2. Соответствие критериям

| Критерий | Реализация |
|---|---|
| Нейронная сеть | MLP с двумя скрытыми слоями по 128 нейронов и ReLU; выход `[batch, n_actions]`; градиенты проверены |
| `epsilon`-greedy | Жадный `argmax`, случайные действия и маска вероятности `epsilon` |
| Взаимодействие | `play_and_record` выполняет точное число шагов; при любом окончании среда перезапускается |
| Vanilla TD-loss | `Q(s,a)` через `gather`; target отделён от графа; маска использует только `terminated` |
| Обучение | Replay mini-batch, Adam, gradient clipping, target update, validation, early stopping, лучший checkpoint |
| Double DQN | Online-сеть выбирает `a*`, target-сеть вычисляет `Q_target(s',a*)` |
| Dueling DQN | `Q = V + A - mean(A)` по размерности действий |
| Dueling против MC | Обратный расчёт discounted return и калибровочный график с осью `x=y` |
| Перенос на LunarLander | Dueling Double DQN, 8 наблюдений, 4 действия, Huber loss |
| Проверка LunarLander | 20 validation-эпизодов, robust selection, 30 независимых финальных seed |
| Аналитика | Кривые обучения, гистограмма, пять эпизодов, траектории, MAE/bias/корреляция, видео и вывод |

## 3. Важные особенности корректности

### `terminated` и `truncated`

В replay сохраняется `done = terminated`, а не логическое `terminated or truncated`. Поэтому при естественном терминальном состоянии bootstrap обнуляется, а при искусственном ограничении `TimeLimit` — сохраняется. Эпизод при этом перезапускается в обоих случаях.

Для сравнения ценности с Monte-Carlo return после `truncated` используется bootstrap-оценка следующего состояния. Это делает MC-сравнение согласованным с TD-целью и снижает систематическое смещение из-за ограничения длины эпизода.

### Независимость оценки

Validation seed фиксированы внутри каждого эксперимента, чтобы сравнение checkpoint было менее шумным. Финальные 30 seed не пересекаются с validation. Пять эпизодов анализа LunarLander и видео используют дополнительные отдельные seed. Видео не рассматривается как статистическая оценка качества.

### Выбор checkpoint

Для CartPole лучший checkpoint выбирается по среднему validation reward. Для LunarLander используется robust-критерий:

```text
selection_score = validation_mean - 0.25 * validation_std
```

Дополнительно вычисляется 10-й percentile. Ранняя остановка LunarLander требует трёх последовательных проверок с `mean >= 250` и `p10 >= 150`. После обучения восстанавливается именно лучший validation checkpoint, а не обязательно последняя сеть.

## 4. Гиперпараметры

| Параметр | CartPole DQN | CartPole Dueling Double | LunarLander Dueling Double |
|---|---:|---:|---:|
| Максимум шагов обучения | 60 000 | 60 000 | 200 000 |
| Replay capacity | 10 000 | 10 000 | 100 000 |
| Начальное заполнение | 10 000 | 10 000 | 10 000 |
| Batch | 64 | 64 | 64 |
| Learning rate | 0.0005 | 0.0005 | 0.0005 |
| `gamma` | 0.99 | 0.99 | 0.99 |
| Epsilon decay | 25 000 | 25 000 | 75 000 |
| Final epsilon | 0.02 | 0.02 | 0.01 |
| Target update | 500 | 500 | 1 000 |
| Validation frequency | 2 000 | 2 000 | 5 000 |
| Loss | MSE | MSE | Huber |
| Gradient norm clip | 10 | 10 | 10 |
| Validation games | 10 | 10 | 20 |
| Final games | 30 | 30 | 30 |

## 5. Фактические результаты

Результаты ниже извлечены из outputs фактического полного `Run All` файла
`MEPHI_RL26_hw2_DQN_Бужор_Александра.ipynb`. Среда выполнения: Python 3.12.16, PyTorch 2.2.2+cpu,
Gymnasium 1.4.0, NumPy 1.26.4.

| Эксперимент | Лучший validation | Финальный mean ± std | p10 | Min / Max |
|---|---:|---:|---:|---:|
| CartPole DQN | 500.0 (шаг 54000) | 500.0 ± 0.0 | — | 500 / 500 |
| CartPole Dueling Double DQN | 500.0 | 500.0 ± 0.0 | — | 500 / 500 |
| LunarLander Dueling Double DQN | 262.0 ± 40.7; p10=239.5 (шаг 170000) | 268.2 ± 35.9 | 239.0 | 155.7 / 320.3 |

Финальная оценка каждого эксперимента выполнена на 30 независимых seed.
Для LunarLander: median=271.0; порог 200 достигнут.

### Пять эпизодов LunarLander

| Эпизод | Return | Шагов |
|---:|---:|---:|
| 1 | 252.6 | 180 |
| 2 | 133.6 | 1000 |
| 3 | 263.6 | 175 |
| 4 | 297.2 | 197 |
| 5 | 244.6 | 231 |

### Калибровка ценности LunarLander

- MAE между оценкой агента и MC return: **22.75**.
- Bias `V_DQN - MC`: **-16.87**; агент систематически недооценивает ценность.
- Корреляция Пирсона: **0.791** (умеренная по абсолютной величине).

Интерпретация построена по mean/std/p10, а не по одному видео. Пять дополнительных
эпизодов дают return [252.6, 133.6, 263.6, 297.2, 244.6] и длины [180, 1000, 175, 197, 231] соответственно.
Все графики и видео сохранены в выполненном ноутбуке. Предварительные проверки:
17/17 автотестов и 48/48 кодовых ячеек smoke `Run All` пройдены; эти smoke-значения
не использованы в таблице результатов.

## 6. Интерпретация графиков

- **Validation reward** — основная метрика качества; для LunarLander важны также std и p10.
- **TD-loss** не обязан монотонно убывать: target меняется при обновлении online- и target-сетей, а распределение replay постоянно сдвигается.
- **Gradient norm** позволяет обнаружить нестабильные обновления; clipping ограничивает влияние выбросов.
- **Initial state value** помогает увидеть рост масштаба Q-оценок, но сам по себе не доказывает улучшение политики.
- **MAE и bias** измеряют абсолютную и знаковую ошибку `V(s)` относительно MC return, а корреляция показывает согласованность ранжирования состояний.
- **Гистограмма и p10** показывают, не держится ли среднее на нескольких удачных эпизодах.

## 7. Воспроизведение

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe train.py --config configs/cartpole_dqn.json
.\.venv\Scripts\python.exe train.py --config configs/cartpole_dueling.json
.\.venv\Scripts\python.exe train.py --config configs/lunarlander_dueling.json
```

Полный ноутбук выполняется через **Restart Kernel and Run All** либо командой `.\.venv\Scripts\python.exe scripts/run_notebook.py`. Smoke-режим проверяет только инфраструктуру и не используется для выводов о качестве агентов.
