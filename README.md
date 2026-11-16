# DQN — от CartPole к LunarLander

Домашняя работа № 2 по обучению с подкреплением.

**Автор:** Александра Бужор  
**Telegram:** [@Angry_C_F_O](https://t.me/Angry_C_F_O)

Проект содержит два равнозначных способа работы:

1. **Самодостаточный ноутбук для сдачи** — `MEPHI_RL26_hw2_DQN_Бужор_Александра.ipynb`. Все ячейки с решением, проверки, обучение, графики, видео и вывод находятся в одном файле и не импортируют локальный пакет.
2. **Модульная версия для VS Code** — пакет `dqn`, JSON-конфигурации, CLI обучения/оценки и автоматические тесты.

## Что реализовано

- MLP DQN и `epsilon`-greedy политика;
- циклический replay buffer и равномерная выборка mini-batch;
- корректное разделение `terminated` и `truncated`: TimeLimit перезапускает эпизод, но не обнуляет bootstrap;
- Vanilla DQN TD-loss с отсоединённой target-целью;
- Double DQN: online-сеть выбирает следующее действие, target-сеть его оценивает;
- Dueling DQN: `Q = V + A - mean(A)`;
- линейное уменьшение `epsilon`, clipping градиента и периодическая синхронизация target-сети;
- validation на фиксированных seed, ранняя остановка и восстановление лучшего checkpoint;
- финальная проверка на 30 новых seed;
- MSE и Huber loss;
- графики reward, TD-loss, нормы градиента и `V(s₀)`;
- Monte-Carlo return, MAE, bias, корреляция и калибровочный график;
- траектории пяти эпизодов LunarLander и MP4-видео;
- unit- и integration-тесты, включая полный короткий цикл обучения.

## Структура

```text
.
├── configs/                         # Полные и smoke-конфигурации
├── dqn/
│   ├── agents.py                    # DQN и Dueling DQN
│   ├── losses.py                    # Vanilla и Double TD-loss
│   ├── replay.py                    # Replay buffer
│   ├── interaction.py               # play_and_record
│   ├── training.py                  # Обучение и checkpoint
│   ├── evaluation.py                # Независимая оценка и MC-return
│   ├── analysis.py                  # Графики и анализ эпизодов
│   ├── checkpoints.py               # Сохранение/загрузка best.pt
│   └── video.py                     # Запись MP4
├── tests/                           # Pytest
├── scripts/                         # Запуск и проверка ноутбука
├── results/                         # Воспроизведённые метрики и артефакты
├── train.py                         # CLI обучения
├── evaluate.py                      # CLI независимой оценки
├── MEPHI_RL26_hw2_DQN_Бужор_Александра.ipynb
└── REPORT.md                        # Методика и фактические результаты
```

## 1. Установка в VS Code (Windows)

Рекомендуется 64-разрядный **Python 3.12**. В терминале VS Code из корня проекта:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Для повторения именно проверенного набора версий можно вместо `requirements-dev.txt` использовать `requirements-lock.txt`, затем выполнить ту же editable-установку с `--no-deps`.

В VS Code нажмите `Ctrl+Shift+P` → **Python: Select Interpreter** → выберите `.venv\Scripts\python.exe`. Настройки запуска и рекомендуемые расширения уже находятся в `.vscode/`.

> Для Linux/macOS команды аналогичны, но интерпретатор находится в `.venv/bin/python`.

## 2. Быстрая проверка

```powershell
.\.venv\Scripts\python.exe scripts/check_setup.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe train.py --config configs/smoke_cartpole.json --skip-video
```

Smoke-режим нужен только для проверки инфраструктуры: 600 шагов недостаточно для хорошей политики. Его результаты сохраняются в игнорируемую Git папку `runs/smoke_cartpole/`.

## 3. Полное обучение

Команды запускаются последовательно:

```powershell
.\.venv\Scripts\python.exe train.py --config configs/cartpole_dqn.json
.\.venv\Scripts\python.exe train.py --config configs/cartpole_dueling.json
.\.venv\Scripts\python.exe train.py --config configs/lunarlander_dueling.json
```

По умолчанию выбирается CUDA, если она доступна, иначе CPU. Для принудительного запуска на CPU добавьте `--device cpu`. Полное обучение занимает заметно больше времени, чем smoke-режим; ранняя остановка срабатывает только после серии сильных validation-результатов.

Полезные сокращения для отладки:

```powershell
.\.venv\Scripts\python.exe train.py --config configs/lunarlander_dueling.json --total-steps 1000 --eval-freq 250 --output-dir runs/lunar_debug --skip-video --skip-analysis
```

Такой короткий запуск является проверкой кода, а не решением задачи обучения.

## 4. Независимая проверка checkpoint

Пример на новых seed, начиная с `500000`:

```powershell
.\.venv\Scripts\python.exe evaluate.py results/cartpole_dqn/best.pt --games 30 --seed 500000
.\.venv\Scripts\python.exe evaluate.py results/lunarlander_dueling_double_dqn/best.pt --games 30 --seed 500000
```

Опция `--video` дополнительно записывает один наглядный эпизод. Видео не используется как статистическая оценка качества.

## 5. Ноутбук для сдачи

1. Откройте `MEPHI_RL26_hw2_DQN_Бужор_Александра.ipynb` в VS Code.
2. Выберите kernel проекта `.venv`.
3. Выполните **Run All** / **Restart Kernel and Run All**.
4. Дождитесь обучения трёх агентов, построения графиков и создания видео.
5. Сохраните ноутбук вместе с outputs и приложите именно его в форму ответа.

Проверка сохранённого выполненного ноутбука:

```powershell
.\.venv\Scripts\python.exe scripts/validate_notebook.py
```

Автоматический полный `Run All` из терминала:

```powershell
.\.venv\Scripts\python.exe scripts/run_notebook.py
```

В Colab ноутбук также самодостаточен: первая ячейка устанавливает отсутствующие зависимости.

## Результаты эксперимента

Полный `Run All` выполнен: **48/48 кодовых ячеек без ошибок**, все **17/17 автотестов** пройдены. На проверочном CPU-запуске обучение, независимая оценка, графики и видео заняли около 50 минут.

| Эксперимент | Финальный mean ± std | p10 | Число независимых эпизодов |
|---|---:|---:|---:|
| CartPole DQN | 500,0 ± 0,0 | — | 30 |
| CartPole Dueling Double DQN | 500,0 ± 0,0 | — | 30 |
| LunarLander Dueling Double DQN | **268,2 ± 35,9** | **239,0** | 30 |

Порог LunarLander 200 достигнут: 28 из 30 эпизодов имеют reward ≥ 200. Лучший checkpoint восстановлен с шага 170 000; его validation-результат — 262,0 ± 40,7, p10 = 239,5. Графики и MP4 встроены в выполненный ноутбук; подробная методика и анализ пяти эпизодов приведены в `REPORT.md`.

Каждая папка `results/<experiment>/` содержит:

- `best.pt` — лучший checkpoint по validation;
- `summary.json` — конфигурация, лучший шаг и финальные метрики;
- `history.json` и `training_curves.png` — история и графики обучения;
- `final_scores.npy` — reward каждого финального эпизода;
- `videos/*.mp4` — наглядный эпизод;
- `analysis/` — пять эпизодов, калибровка и траектории, если включено в конфигурации.

Сводка фактических запусков и выводы находятся в `REPORT.md`. Числа не следует переносить из smoke-режима в итоговый отчёт.

## Ключевые формулы

Vanilla DQN:

```text
y = r + gamma * (1 - terminated) * max_a Q_target(s', a)
L = MSE(Q_online(s, a), stop_gradient(y))
```

Double DQN:

```text
a* = argmax_a Q_online(s', a)
y = r + gamma * (1 - terminated) * Q_target(s', a*)
```

Dueling DQN:

```text
Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')
```

Для LunarLander выбраны Dueling Double DQN и Huber loss. Checkpoint сравнивается по `validation_mean - 0.25 * validation_std`; дополнительно отслеживается 10-й percentile reward.

## Воспроизводимость и seed

- Обучение: seed из JSON (`42` для DQN и LunarLander, `142` для Dueling CartPole).
- Validation: фиксированный набор seed для сопоставимого выбора checkpoint.
- Final: отдельный диапазон, не пересекающийся с validation.
- Анализ и видео: отдельные seed.
- Результаты на разных версиях библиотек и аппаратном обеспечении могут немного отличаться, поэтому вместе с кодом сохраняются конфигурация и метрики фактического запуска.

## Публикация на GitHub

Создайте пустой репозиторий на GitHub без README, затем из корня проекта:

```powershell
git init
git add .
git commit -m "Implement DQN, Double DQN and Dueling DQN"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/dqn-cartpole-lunarlander.git
git push -u origin main
```

`.gitignore` исключает `.venv`, кэши, исходные вложения и временные `runs`. Выполненный ноутбук и папка `results` предназначены для публикации. Перед push проверьте размер файлов командой `git status` и историю больших артефактов; GitHub ограничивает отдельные файлы размером 100 МБ.

## Если Box2D не установился

1. Убедитесь, что используется 64-разрядный Python 3.12 и обновлённый `pip`.
2. Повторите `.\.venv\Scripts\python.exe -m pip install --force-reinstall "gymnasium[box2d]>=1.1,<2"`.
3. Если вместо готового wheel началась сборка из исходников, установите Microsoft C++ Build Tools и повторите команду. `swig` уже указан в зависимостях.
4. Проверьте среду отдельно:

```powershell
.\.venv\Scripts\python.exe -c "import gymnasium as gym; e=gym.make('LunarLander-v3'); print(e.reset(seed=0)[0].shape); e.close()"
```
