# System Punktacji - Dokumentacja

## Przegląd

Całkowity score składa się z 3 głównych komponentów, które są agregowane w wynik końcowy (0-100 punktów).

```
OVERALL_SCORE =
    (docker_component × 0.35) +
    (k8s_component   × 0.40) +
    (runtime_phase   × 0.25)
```

---

## 1. DOCKER COMPONENT (35% wagi całkowitej)

Komponent Docker składa się z 3 faz walidacji:

### Fazy Docker:

| Faza | Waga | Opis |
|------|------|------|
| **DOCKER_SYNTAX** | 40% | Walidacja składni Dockerfile - sprawdza czy Dockerfile jest poprawny składniowo |
| **DOCKER_BUILD** | 40% | Walidacja budowania obrazu - sprawdza czy `docker build` się powiedzie |
| **DOCKER_LINTERS** | 20% | Walidacja przez lintery (np. hadolint) - sprawdza najlepsze praktyki Dockerfile |

### Specjalne reguły:

⚠️ **Reguła krytyczna**: Jeśli `docker_build` ma błędy (ERROR), **wszystkie** fazy Docker dostają 0 punktów

**Lokalizacja w kodzie**: `src/evaluator/core/scoring_model.py:221-224`

---

## 2. KUBERNETES COMPONENT (40% wagi całkowitej)

Komponent Kubernetes składa się z 3 faz walidacji:

### Fazy Kubernetes:

| Faza | Waga | Opis |
|------|------|------|
| **K8S_SYNTAX** | 40% | Walidacja składni plików Kubernetes YAML - sprawdza czy manifesty są poprawne składniowo |
| **KUBERNETES_APPLY** | 40% | Walidacja możliwości wdrożenia na klaster - sprawdza czy `kubectl apply` się powiedzie |
| **K8S_LINTERS** | 20% | Walidacja przez lintery (np. kubeval, kube-linter) - sprawdza najlepsze praktyki i potencjalne problemy |

### Specjalne reguły:

⚠️ **Reguła krytyczna**: Jeśli `kubernetes_apply` ma błędy (ERROR), **wszystkie** fazy Kubernetes oraz Runtime dostają 0 punktów

**Lokalizacja w kodzie**: `src/evaluator/core/scoring_model.py:227-233`

---

## 3. RUNTIME COMPONENT (25% wagi całkowitej)

Komponent Runtime to walidacja działania aplikacji w środowisku Kubernetes:

### Charakterystyka:

- **Binarna ocena**:
  - Sukces = 100 punktów
  - Failure = 0 punktów
- **Kara za ostrzeżenia**: Każde WARNING odejmuje 10 punktów od wyniku
- **Zależność**: Runtime score zależy od powodzenia poprzednich faz (zwłaszcza `kubernetes_apply`)

**Lokalizacja w kodzie**: `src/evaluator/core/scoring_model.py:339-369`

---

## System Kar za Błędy

Każda faza (oprócz runtime) zaczyna od **100 punktów** i traci punkty według następujących reguł:

```python
BASE_PHASE_SCORE = 100.0

# Kary za każde wystąpienie:
ERROR:   -15 punktów
WARNING: -10 punktów
INFO:     -0 punktów (tylko informacyjne)

# Obliczenie końcowego wyniku fazy:
final_score = max(0, 100 - suma_kar)
```

### Przykład:

Jeśli faza ma:
- 2 błędy (ERROR)
- 3 ostrzeżenia (WARNING)
- 5 informacji (INFO)

To:
```
Kary = (2 × 15) + (3 × 10) + (5 × 0) = 30 + 30 + 0 = 60 punktów
Final Score = max(0, 100 - 60) = 40 punktów
```

**Lokalizacja w kodzie**: `src/evaluator/core/scoring_model.py:146-151, 268-297`

---

## Agregacja Komponentów

### Wagi komponentów:

| Komponent | Waga | Uzasadnienie |
|-----------|------|--------------|
| Docker | 35% | Podstawowa konfiguracja konteneryzacji |
| Kubernetes | 40% | Kluczowa konfiguracja wdrożenia (najważniejsza) |
| Runtime | 25% | Walidacja działania (zależy od poprzednich faz) |

### Brak normalizacji wag:

⚠️ **WAŻNE**: Wagi NIE są normalizowane. Jeśli jakiś komponent **nie został uruchomiony**, jest traktowany jako **0 punktów**.

**Przykład**: Jeśli runtime nie został uruchomiony:
```
docker_score = 100
k8s_score = 100
runtime_score = 0 (nie uruchomiony)

overall_score = 100×0.35 + 100×0.40 + 0×0.25 = 75 punktów
```

To oznacza, że aby dostać maksymalny wynik, **musisz przejść przez wszystkie fazy**.

**Lokalizacja w kodzie**: `src/evaluator/core/scoring_model.py:410-432`

---

## Diagram Przepływu Punktacji

```
┌─────────────────────────────────────────────────────────────┐
│                      OVERALL SCORE (0-100)                  │
│                                                             │
│  = docker(35%) + kubernetes(40%) + runtime(25%)            │
└─────────────────────────────────────────────────────────────┘
           ▲                ▲                ▲
           │                │                │
┌──────────┴──────┐  ┌──────┴──────┐  ┌─────┴─────┐
│  Docker (35%)   │  │  K8s (40%)  │  │Runtime(25%)│
│                 │  │             │  │            │
│ ┌─────────────┐ │  │┌──────────┐│  │┌──────────┐│
│ │Syntax  (40%)│ │  ││Syntax(35%)││  ││ Runtime  ││
│ └─────────────┘ │  │└──────────┘│  │└──────────┘│
│                 │  │             │  │            │
│ ┌─────────────┐ │  │┌──────────┐│  │ Binary:    │
│ │Build   (40%)│ │  ││Apply(40%) ││  │ Success=100│
│ └─────────────┘ │  │└──────────┘│  │ Fail=0     │
│                 │  │             │  │ -10/warning│
│ ┌─────────────┐ │  │┌──────────┐│  │            │
│ │Linters (20%)│ │  ││Lint (25%) ││  │            │
│ └─────────────┘ │  │└──────────┘│  │            │
└─────────────────┘  └─────────────┘  └────────────┘
```

---

## Łańcuch Zależności

```
docker_build [ERROR]
    ↓
    → Wszystkie fazy Docker = 0

kubernetes_apply [ERROR]
    ↓
    → Wszystkie fazy Kubernetes = 0
    → Runtime = 0
```

---

## Struktura Danych

### AggregatedScore

Główna struktura wyniku zawierająca:

```python
@dataclass
class AggregatedScore:
    overall_score: float              # Wynik końcowy (0-100)
    docker_component: ComponentScore  # Wynik Docker z fazami
    k8s_component: ComponentScore     # Wynik K8s z fazami
    runtime_score: PhaseScore         # Wynik Runtime
    total_errors: int                 # Suma wszystkich błędów
    total_warnings: int               # Suma wszystkich ostrzeżeń
    total_info: int                   # Suma wszystkich info
```

### ComponentScore

Wynik dla komponentu (Docker/Kubernetes):

```python
@dataclass
class ComponentScore:
    component_name: str               # "Docker" lub "Kubernetes"
    phase_scores: List[PhaseScore]    # Lista wyników faz
    weighted_score: float             # Ważona średnia faz
    total_errors: int                 # Suma błędów w komponencie
    total_warnings: int               # Suma ostrzeżeń w komponencie
    total_info: int                   # Suma info w komponencie
```

### PhaseScore

Wynik dla pojedynczej fazy:

```python
@dataclass
class PhaseScore:
    phase: ValidationPhase            # Typ fazy
    base_score: float = 100.0         # Wynik bazowy
    error_count: int                  # Liczba błędów
    warning_count: int                # Liczba ostrzeżeń
    info_count: int                   # Liczba info
    error_penalty: float              # Kara za błędy (count × 15)
    warning_penalty: float            # Kara za ostrzeżenia (count × 10)
    info_penalty: float               # Kara za info (count × 0)
    final_score: float                # Wynik końcowy fazy
    issues: List[ValidationIssue]     # Lista wszystkich issue
```

---

## Konfiguracja

Wszystkie wagi i kary są zdefiniowane w klasie `ScoringConfig`:

**Lokalizacja**: `src/evaluator/core/scoring_model.py:139-184`

### Możliwość dostosowania:

Można utworzyć własną konfigurację poprzez:

```python
from src.evaluator.core.scoring_model import ScoringConfig, IssueAggregationModel

# Własna konfiguracja
config = ScoringConfig()
config.SEVERITY_PENALTIES[ValidationSeverity.ERROR] = 20.0  # Zwiększ karę za błędy
config.COMPONENT_WEIGHTS["kubernetes"] = 0.50               # Zwiększ wagę K8s

# Użycie własnej konfiguracji
model = IssueAggregationModel(config=config)
score = model.calculate_scores(step_issues, runtime_success)
```

---

## Przykłady Obliczeń

### Przykład 1: Pełny sukces

```
Docker:
  - syntax: 100 (0 issues)
  - build: 100 (0 issues)
  - linters: 100 (0 issues)
  → docker_component = (100×0.4 + 100×0.4 + 100×0.2) = 100

K8s:
  - syntax: 100 (0 issues)
  - apply: 100 (0 issues)
  - linters: 100 (0 issues)
  → k8s_component = (100×0.4 + 100×0.4 + 100×0.2) = 100

Runtime: 100 (success, 0 warnings)

OVERALL = 100×0.35 + 100×0.4 + 100×0.25 = 100
```

### Przykład 2: Z błędami

```
Docker:
  - syntax: 85 (1 error: -15, 0 warnings)
  - build: 70 (2 errors: -30, 0 warnings)
  - linters: 80 (2 warnings: -20)
  → docker_component = (85×0.4 + 70×0.4 + 80×0.2) = 78

K8s:
  - syntax: 100 (0 issues)
  - apply: 100 (0 issues)
  - linters: 90 (1 warning: -10)
  → k8s_component = (100×0.4 + 100×0.4 + 90×0.2) = 98

Runtime: 80 (success, 2 warnings: -20)

OVERALL = 78×0.35 + 98×0.4 + 80×0.25 = 86.5
```

### Przykład 3: Build failure

```
Docker:
  - syntax: 100 (0 issues)
  - build: 55 (3 errors: -45, 0 warnings) → HAS ERRORS!
  - linters: 90 (1 warning: -10)

⚠️ Build ma błędy → wszystkie fazy Docker = 0
  → docker_component = 0

K8s:
  - syntax: 100
  - apply: 100
  - linters: 100
  → k8s_component = 100

Runtime: 100

OVERALL = 0×0.35 + 100×0.4 + 100×0.25 = 65
```

---

## Źródło

Cała logika znajduje się w:
- **Główny plik**: `src/evaluator/core/scoring_model.py`
- **Użycie**: `src/evaluator/experiments/runner.py:433`
- **UI Dashboard**: `ui/experiment_dashboard.py:124`

---

## Zasady Projektowe

1. **Phase-based scoring**: Różne fazy walidacji mają różne wagi
2. **Severity-based penalties**: Błędy mają różny wpływ w zależności od severity
3. **Reproducible**: Deterministyczne wagi zapewniają spójność między uruchomieniami
4. **Transparent**: Szczegółowe rozbicie per-faza umożliwia debugging i analizę
5. **Weighted aggregation**: Krytyczne fazy mają większy wpływ na wynik końcowy
