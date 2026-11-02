Статья [[Parting with Illusions about Deep Active Learning.pdf|Parting with Illusions about Deep Active Learning]] ставит под сомнение многие бенчмарки, которые проводили авторы стратегий AL. 
Авторы провели собственные бенчмарки для методов, основанных на неопределенности и diversity.
Главные тезисы:
1. Аугментация съедает разницу между AL и random.
После добавления аугментаций так же меняется ранжирование в метриках точности внутри стратегий AL.

Сравнение AL без аугментаций и с аугментациями:
![[Pasted image 20260522143421.png]]

Так происходит потому что с помощью аугментаций (CutMix/Copy-Paste/Mixup) мы итак генерируем сложные примеры, которые уже идут на обучение.

2. SSL (semi-supervised) дает больший прирост, чем AL.
Random strategy + SSL > AL. Лучше использовать для обучения unlabeled pool, а не только для выбора разметки. Можно попробовать объединить AL + SSL.

Сравнение AL и AL + SSL на CIFAR-10 и CIFAR-100:
![[Pasted image 20260522143310.png]]

SSL эффективно использует новые неразмеченные данные для своего дообучения. Фактически при обучении с AL есть две сущности:
- Labeled pool
- Unlabeled pool

AL для обучения использует только labeled множество. Unlabeled использует для прогонов через модель просто чтобы выбрать что разметить дальше. После выбора unlabeled выбрасывается из вычислений до следующего цикла. Фактически модель получает gradient signal только из 5000 примеров.

SSL для обучения использует оба множества. Для labeled используется стандартный supervised loss, из unlabeled -- consistency loss, pseudo-labeling и др. сигналы. Каждый unlabeled пример участвует в каждом forward/backward проходе.

3. В low-budget режимах AL проигрывает random. 
   ![[Pasted image 20260522143624.png]]


4. Ранжирование методов нестабильно между датасетами.
У авторов порядок методов по качеству менялся в зависимости от датасета (CIFAR-10 или CIFAR-100).

Гипотеза о причине проигрыша Active Learning у random стратегии.
Active Learning вводит selection bias в обучающую выборку. Распределение выбранных примеров не совпадает с истинным распределением данных. Вред от этого сдвига превышает пользу от "более информативных" примеров. 

# Анализ результатов Active Learning
Результаты Active Learning для семантической сегментации рельсов
Общий датасет состоял из 50% человеческой разметки на рельсах Испании и 50% псевдо-разметки от DinoV3-SAT493M на ортофотоплане Лужской (всего изображений 3960). Подробный тех. репорт о пайплайне см. в [[/home/samsepi/Documents/code/active_learning/plots_comparison/EXPERIMENT_REPORT.md]] 
Обучение начинается в 6 циклов AL. Старт с бюджетом 5% от общего размера датасета -- 198 картинок отобранных случайным образом (seed фиксирован для всех стратегий), с каждым циклом добавляется 5%, в конце доходим до 30%.

## Test-set bars
### 1. `test_bars_IoU.png` — Test IoU, ranked

![[test_bars_IoU 1.png]]

**Вывод:** BALD (0.8415) и BvSB (0.8391) **обгоняют fulldata** (0.8386) на 30% данных; gap внутри стандартного отклонения, но это уже паритет с верхней границей. Coreset (0.808) — единственный проигравший random.


### 2. `test_bars_F1.png` — Test F1, ranked

![[test_bars_F1.png]]


### 3. `test_bars_clDice.png` — Test clDice, ranked

![[test_bars_clDice.png]]

**Вывод:** BALD (0.9535) и BvSB (0.9520) **превосходят fulldata** (0.9467) на топологической метрике на 0.5-0.7 pp — самый чистый результат, ошибки баров не пересекаются. **Для тонких рельсовых линий AL топологически лучше, чем full data.**


### 4. `test_bars_Loss.png` — Test Loss, ranked

![[test_bars_Loss.png]]

**Вывод:** Loss инвертирует random — он **последний** (худший), даже хуже coreset. BALD/BvSB/margin показывают значимо более низкий loss, чем fulldata → модели не только точнее, но и более калиброванные.

## Delta vs random
### 5. `test_delta_vs_random_IoU.png`

![[test_delta_vs_random_IoU.png]]

**Вывод:** BALD +2.44, BvSB +2.20, margin +1.62 pp — **3 стратегии обгоняют random на величину больше σ**. Coreset единственный **отрицательный** (−0.92 pp). Линия fulldata (+2.14) ниже BALD и BvSB.

### 6. `test_delta_vs_random_F1.png`

![[test_delta_vs_random_F1.png]]


### 7. `test_delta_vs_random_clDice.png`

![[test_delta_vs_random_clDice.png]]

**Вывод:** Coreset (−0.72) И entropy (−0.30) **ниже random** на топологической метрике. Margin/BvSB/BALD выше. То есть **entropy подозрительно**: формально IoU чуть лучше random, но топологически хуже.

### 8. `test_delta_vs_random_Loss.png`

![[test_delta_vs_random_Loss.png]]

**Вывод:** Все стратегии лучше random по loss; даже coreset (+0.0014). Random — **самый плохо калиброванный baseline**. Loss-метрика более снисходительна к coreset, чем IoU/clDice.

## Delta vs fulldata (4 графика) — "догнали ли upper bound"

### 9. `test_delta_vs_fulldata_IoU.png`

![[test_delta_vs_fulldata_IoU.png]]

**Вывод:** Только **BALD (+0.0029) и BvSB (+0.0005) положительны** — буквально достигли или превысили fulldata. Coreset на −3 pp ниже upper bound, что фактически равно "учиться на чистом random sampling из 30% данных без AL".

### 10. `test_delta_vs_fulldata_F1.png`

![[test_delta_vs_fulldata_F1.png]]

### 11. `test_delta_vs_fulldata_clDice.png`

![[test_delta_vs_fulldata_clDice.png]]

**Вывод:** Здесь запас больше — BALD +0.68 pp, BvSB +0.53 pp. Самый сильный аргумент работы: **AL превышает fulldata именно на структурно-важной метрике для рельсов**.

### 12. `test_delta_vs_fulldata_Loss.png`

![[test_delta_vs_fulldata_Loss 1.png]]


## Val curves per cycle

### 13. `val_curves_IoU.png`

![[val_curves_IoU.png]]

**Вывод:** Margin/BvSB **плато c cycle 1** (10% данных), дальше горизонтально. BALD/coreset_only_old растут **монотонно** до cycle 5.

### 14. `val_curves_F1.png`

![[val_curves_F1.png]]

### 15. `val_curves_clDice.png`

![[val_curves_clDice.png]]


### 16. `val_curves_Loss.png`

![[val_curves_Loss.png]]

### 17. `val_per_strategy_IoU.png`

![[val_per_strategy_IoU.png]]

### 18. `val_per_strategy_clDice.png`
![[val_per_strategy_clDice.png]]

## Pool composition

### 19. `picks_composition.png`

![[picks_composition.png]]

**Вывод:** Coreset берёт **≥50% real** (выше pool ratio), все uncertainty-методы берут **20-32% real** (ниже pool ratio). Random держится 50%. Это **визуализация selection bias direction** — coreset смещается К шуму, uncertainty ОТ шума. 

## Ablation

### 20. `coreset_ablation.png`

![[coreset_ablation.png]]

**Вывод:** `coreset_only_old` бьёт обычный coreset на **+2.22 IoU / +1.38 F1 / +1.00 clDice** при том же бюджете. **Подтверждает гипотезу:** проигрыш coreset не от k-center алгоритма, а от того, что k-center ловит выбросы. 