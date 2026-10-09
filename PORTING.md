# Porting PRIMs from Swift to Python

Status, 4 October 2026: the simulation core is ported and validated against
Swift batch output. Two things are not ported: the GUI and the obsolete
`screen`/`transition` definitions. This file has four parts:

1. the module map;
2. Swift behaviours kept on purpose;
3. places where the Python version deliberately differs;
4. what the validation shows.

Source: `PRIMs/Core/*.swift` (about 7,500 lines; the full app is about
10,000 lines including the SwiftUI views). The Python package is about
4,100 lines (about 3,200 lines of code; the rest is docstrings, comments and blank lines).


The embedded representation (`pyprims/embedding.py`, `EMBEDDINGS.md`) is an
extension, not part of the port: the default symbolic representation is
the port described here, unchanged.

## 1. Module map

| Swift | lines | Python | notes |
|---|---:|---|---|
| Core/Support.swift | 126 | `values.py` | `Value` → Chunk / float / str; equality by description; `string2Double`; `actrNoise`; `chopPrims` |
| Core/Chunk.swift | 446 | `chunk.py` | base level (ol and full), Sji, spreading, noise cached per time step |
| Core/Declarative.swift | 566 | `declarative.py` | add/merge, retrieve, partial matching, blending, retrieval module, RT push/pop |
| Core/Prim.swift | 316 | `prim.py` | `parseName` (incl. canonical order of comparisons), `Prim.fire` |
| Core/Procedural.swift, Production.swift, Instantiation.swift | 433 | `procedural.py` | task-general productions, utility learning, production compilation |
| Core/Operator.swift | 674 | `operators.py` | operator retrieval and matching, execution timing, operator compilation (`opc:`), skill/context/inter-operator association learning, bindings |
| Core/Imaginal.swift, Action.swift, Temporal.swift | 567 | `modules.py` | WM trees (push/pop), WM→DM, actions with latency noise, visicon, `build-goal`, interval timer |
| Core/Model.swift, PRScenario.swift, Task.swift | 986 | `model.py` | parameters, tasks, trials, `newStep` (the operator cycle), module actions |
| Core/Parser.swift | 910 | `dsl.py` | one ordered list of definitions, replayed exactly as the Swift parser builds DM; the same list comes from the Python `Task` API or from the .prims reader |
| Core/Script.swift | 1145 | `script_translate.py` | **translator, not interpreter**: the Swift tokenizer and parser, emitting Python |
| Core/ScriptFunctions.swift | 1033 | `script_api.py` | all 51 script functions, as methods of `m` |
| Core/BatchRun.swift | 255 | `batch.py` | same output format; `save-image`/`load-image` use pickle |
| Core/ConflictSetTrace.swift | 186 | — | GUI only; trace level 5 lists the conflict sets |
| Core/GoalOperatorLearning.swift | 43 | — | experimental `altActivation`, not part of the Swift build |
| Views/*, PRIMsViewModel, ModelS, FruchtermanReingold, NSScanner+Swift | ~2,500 | — | SwiftUI interface; not ported |
| — | | `convert.py`, `__main__.py`, `tools/compare_with_swift.py` | new: .prims → .py converter, command line, validation tool |

## 2. Swift behaviours kept on purpose

Each of these is reproduced exactly so that results match. Several look
unintended, and they are worth a decision in both implementations:

1. **Values compare by their printed form.** The number 3.0 and the text "3"
   are different values.
2. **`string2Double` reads a numeric prefix**, so a chunk name like `2nd`
   becomes the number 2.0 when stored in a slot.
3. **Fan** counts the chunks a chunk appears in, *without* the chunk itself
   (ACT-R counts 1 + appearances).
   - Fan is not updated by `stringsToChunks`, so it depends on definition
     order: in `count.prims` the operators mention `count-fact` before the
     facts define it.
   - The Sji fallback is max(0, mas − ln fan).
4. **Spreading from the goal is not divided by the number of goal slots.**
   Every other buffer gives W/n per source.
5. **One-letter constants are not recognised.** `q -> AC2` becomes an
   unparseable PRIM that silently always fails.
   - This is why the copy of `paired.prims` in the root of the tutorial
     repository never answers.
   - Python behaves the same but warns at load time (`ParseWarning`).
6. **`new-pm-exp:` sets the `new-pm-pow` field.**
7. **Operator compilation reads only one digit of a slot name**
   (`Op(chunk:)`), so it is wrong for operators with more than nine
   constants.
8. **Rolling back a failed operator restores the goal to the saved copy
   itself, and the imaginal chunk by reference.**
   - If a second candidate also fails in the same step, the first rollback
     copy has already been modified.
   - Slot writes into an existing WM chunk survive the failure.
9. **`initializeNextTrial` does not clear pending module actions.** If a
   trial ends between the two half-steps, the next trial starts with the
   previous trial's module actions.
10. **Script grammar:**
    - binary operators are right-associative: `a - b - c` is `a - (b - c)`,
      and `a / b * c` is `a / (b * c)`;
    - each comparison needs its own parentheses: `((a == 1) && (b == 2))`;
    - identifiers may contain `-` and `+`, so `i+1` is one name;
    - integer `/` truncates;
    - a string on the left of `+` concatenates.

    Also:

    - `&&` multiplies and `||` adds, and both sides are always evaluated;
    - an `if` treats any non-integer value as true (even `0.0` and `""`);
    - a `while` is entered when its test is non-zero but repeats only while
      the test equals exactly 1;
    - `for i in a to b` re-tests `i <= b` every pass and honours changes to
      `i` in the body;
    - `for x in arr` fixes the number of passes at the start;
    - `<` and `>` are false for strings.

    The translator keeps all of this. It uses plain Python where a small
    type inference shows the meaning is the same, and helper functions
    (`div`, `plus`, `setitem`, `and_`, `or_`, `truth`, `cmp`) or explicit
    loop forms where it is not.
11. **With `explore-exploit:` > 0, retrieval evaluates a chunk's activation
    three times**, each with fresh Sji noise: once for the conflict set,
    once for the comparison, once for the latency.

## 3. Deliberate differences

- **Scripts run straight through.** A script is a Python function, and
  `m.run_until_action(...)` returns when its condition is met. Swift
  suspends its interpreter at each run call so the GUI can single-step;
  stepping can be added later with a generator if a GUI needs it.
- **Falling through.** If no operator matches, the rest of the script still
  runs in the same trial. In Swift batch mode it runs at the start of the
  next trial, and that trial's output then repeats the previous lines,
  because `outputData` is not reset.
- **`time-threshold:` is enforced in batch runs** and logged as
  `trial-end timeout`. Swift checks it only in the GUI, so a stuck model
  hangs a Swift batch.
- **A script that keeps calling run functions on a stuck model raises
  `ModelStuck`.** Swift would loop forever; the tutorial's `add-fingers`
  template does this.
- **No call to `actrNoise` is made when `explore-exploit:` is 0.** The
  distribution is identical; only the random stream differs.
- **Dictionaries iterate in insertion order**, not Swift's hash order. This
  only matters for exact ties, which noise makes vanishingly rare.
- **Variables first assigned inside an if/while/for block stay visible after
  it.** In Swift they are local to the block, so a later use is a runtime
  error and that statement is skipped.
- **A quoted annotation in an operator body** (`"note"`) is skipped as one
  token. Swift skips to the *next* quote character, which can swallow
  PRIMs.
- **Blended retrieval with no match above threshold** returns a retrieval
  failure; Swift crashes on a force-unwrap.
- **Random numbers come from Python's `random.Random(seed)`**, so runs are
  reproducible. They are not stream-identical to `arc4random`.
- **`save-image`/`load-image` use pickle (`.pybrain`).** Pickle cannot store
  scripts that were translated from .prims; re-load the task after loading
  an image.

## 4. Validation

**Exact checks** (`tests/`, 13 tests, all passing):

- PRIM parsing and canonical ordering;
- operator body compilation (constants, `->*binding` targets);
- script semantics (right associativity, truncating division, array
  padding, Swift's loop and truth rules);
- the PRIM firing times of the first operator in the tutorial's count
  trace (0.00, 0.05, 0.35 … 1.55 s);
- the Python `Task` API and the converter both give results identical to
  the .prims reader.

**All 36 tutorial models** load, run, and give identical runs whether loaded
from `.prims` or from the converted `.py`.

**Statistical checks against Swift batch output** shipped with the tutorial
(`examples/swift_reference/`; Swift ran 10 model runs, Python 30–100):

| batch | what is compared | result |
|---|---|---|
| `testbatch` (count → semantic transfer; semantic control) | mean trial time per block of 20, 15 blocks | 14/15 blocks within \|z\| < 2; transfer reproduced (semantic first 20 trials: Swift 4.43 s after count training vs 9.79 s control; Python 4.39 vs 9.82) |
| | count trials 21–40 | Python 4.91–4.99 s vs Swift 4.80 ± 0.05 s (z ≈ 1.9–2.7 in two independent Python runs) — **open** |
| `taskswitchstrooptransfer` | Stroop latency by condition and block (16 cells) | all \|z\| < 2.2; event counts equal (e.g. 4000/4000 say, 3994/3994 attend) |
| | runs that switch to the task-switching `attend-color` operator in Stroop | Swift 9/10, Python 47/60 (Fisher p ≈ 0.7) |

**The one open point.**

- The 10 Swift runs give a per-run SD of 0.16 s on count trials 21–40, where
  200 Python runs give 0.28 s, so the Swift SE is probably too small. About
  5% of random 10-run subsets of Python runs are as fast as the Swift mean.
- The difference sits in the first and last operators (`start-count`,
  `final`), not in `iterate`. `iterate` uses the same production and utility
  code, which argues against a general fault in compilation or utility
  learning.
- An independent line-by-line audit of the simulation core against the
  Swift code found no semantic difference on the paths this model uses.
- The reference `.dat` in the tutorial repository may also come from an
  older Swift version.

The way to settle it:

1. Run `examples/testbatch.bprims` with `repeat 50` in the current Swift
   build.
2. Compare with
   `python tools/compare_with_swift.py SWIFT.dat PYTHON.dat`.

**Exact trace comparison with noise.** This would need the same random
stream in both languages *and* the same chunk iteration order in Swift. The
cheaper exact test is to switch noise off (`ans: 0`, `egs: 0`, actions
without noise, fixed script inputs) and compare traces line by line.

**Speed** (one core, CPython 3.12):

- count: 1,000 trials in 0.7 s;
- the 10-run Stroop/task-switching batch: 46 s.

Retrieval (activation of every matching chunk) takes about half of the time.
Vectorising activation with numpy is the obvious first step if larger models
need it.
