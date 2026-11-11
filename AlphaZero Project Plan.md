# AlphaZero on a MacBook Pro M5: 30-Day Project Plan

This is a very doable project. The most important decision is **which game to target**, so here is that first.

## The big decision: chess, or something smaller first?

AlphaZero's chess run used 5,000 TPUs to generate about 44 million self-play games. On one laptop over roughly two weeks of training time, you will generate somewhere in the tens of thousands of games. That gap is about three orders of magnitude. It doesn't make chess pointless, but it does mean a pure "zero knowledge" chess agent will finish the month weak by human standards.

The recommendation is to build the whole system game-agnostic. Get it working and debugged on Connect Four, where you can reach near-perfect play and check yourself against a known solver. Then port it to chess for the second half of the month. You end up with one strong result and one honest, measurable chess result, and almost all the bugs get found on the cheap game.

## Timeline

The plan assumes about 3–4 focused hours per day, plus overnight compute.

### Days 1–3: MCTS foundations

**Read:** The AlphaGo Zero paper (Silver et al., 2017, *Nature*, "Mastering the game of Go without human knowledge"). The AlphaZero *Science* paper (Silver et al., 2018), and especially its supplementary materials, which contain Python-style pseudocode that is the best single implementation reference. For MCTS background, Kocsis & Szepesvári (2006), "Bandit based Monte-Carlo Planning" (the UCT paper), and sections 1–3 of Browne et al. (2012), "A Survey of Monte Carlo Tree Search Methods."

**Build:** A game interface (legal moves, apply move, terminal check, result, state encoding) and plain UCT with random rollouts. Test it on tic-tac-toe, then on Connect Four.

**Checkpoint:** UCT never loses at tic-tac-toe, and it beats a random player at Connect Four almost every game.

### Days 4–7: The full AlphaZero loop on Connect Four

**Read:** He et al. (2016), "Deep Residual Learning," to refresh the ResNet details. Rosin (2011), "Multi-armed bandits with episode context," the origin of PUCT. Re-read the AlphaZero pseudocode section on PUCT, Dirichlet noise, and temperature.

**Documentation:** The PyTorch MPS backend notes and the `torch.nn` / `torch.optim` basics you need. MLX is the Apple-native alternative and is good, but PyTorch has far more AlphaZero material online.

**Build:** A small dual-head ResNet (policy and value), PUCT search guided by the network, a self-play generator, a replay buffer, and a training step with the AlphaZero loss: value MSE plus policy cross-entropy plus L2. Wire it into the loop: play games, train, repeat.

**Checkpoint:** The loop runs end to end unattended, and the trained network plus 100 simulations beats your days 1–3 UCT with 1,000 rollouts.

### Days 8–11: Make it fast and correct

**Read:** Wu (2019), "Accelerating Self-Play Learning in Go" (the KataGo paper). It is the most important paper for you after AlphaZero itself, because it is all about getting strong results on limited compute. Playout cap randomization and the auxiliary targets are the ideas to note. Also skim Anthony et al. (2017), "Thinking Fast and Slow with Deep Learning and Tree Search" (Expert Iteration), for a cleaner theoretical framing of the same idea.

**Documentation:** Python's `multiprocessing` docs, and Pascal Pons' Connect Four solver write-up, which you can use as ground truth.

**Build:** The key engineering problem is that sending one position at a time to the GPU is hopelessly slow. Run many games in parallel and batch their leaf evaluations into one network call. Add virtual loss if you parallelise within a single tree. Also add horizontal-mirror data augmentation, a fixed evaluation harness (your agent against reference opponents, N games, with win/draw/loss and Elo with confidence intervals), and logging of loss curves, game lengths, and policy entropy.

**Checkpoint:** A large throughput improvement over day 7 (aim for at least 10x in positions per second), and the Connect Four agent picks solver-optimal moves on a large majority of a held-out set of positions. This is the "I have built AlphaZero" milestone.

### Days 12–16: Port to chess

**Read:** Re-read the AlphaZero supplementary material on chess input planes (8×8×119) and the 8×8×73 = 4,672 move encoding.

**Documentation:** The python-chess docs (use it for rules, since writing a fast move generator yourself would eat a week), the Leela Chess Zero (Lc0) project docs and blog for practical encoding notes, the Chess Programming Wiki for background, and the UCI protocol spec.

**Build:** Board-to-tensor encoding (you can simplify the 119 planes heavily, for example with less history), the move-to-index mapping in both directions, legal-move masking on the policy head, game-length caps and draw adjudication, and a minimal UCI wrapper so external tools can play your engine.

**Checkpoint:** Unit tests show that encoding and decoding round-trip correctly for every legal move across thousands of random positions. Self-play produces legal games with no crashes. Bugs in move encoding are the most common way these projects silently fail, so test this hard.

### Days 17–18: Plan the compute run

**Read:** Danihelka et al. (2022), "Policy improvement by planning with Gumbel" (Gumbel AlphaZero/MuZero). It gives good policy improvement with very few simulations (16–32 instead of 800), which matters a lot on a laptop.

**Build:** Benchmark games per hour at your chosen settings, then choose network size and simulation count to fit your remaining days. A reasonable starting point is 6–10 residual blocks, 64–128 channels, and 100–200 PUCT simulations (or Gumbel with 16–32). Also decide whether to add a supervised bootstrap (see the Elo section below).

**Checkpoint:** A written compute budget and a configuration you have watched run cleanly for at least an hour.

### Days 19–26: The long training run

**Read:** No new required reading. Optionally, the MuZero paper (Schrittwieser et al., 2020).

**Build:** Train continuously, overnight included. Keep the MacBook plugged in and somewhere cool, since sustained load will cause some thermal throttling. Every day, save a checkpoint and run a short match against the previous one.

**Checkpoint:** Each daily checkpoint beats the previous day's with statistical confidence. If progress stalls for two days, stop and debug rather than burning compute.

### Days 27–29: Evaluation

**Documentation:** The cutechess-cli docs (for running engine matches), Ordo or BayesElo (for computing ratings), and Stockfish's `UCI_LimitStrength` / `UCI_Elo` options.

**Build:** Run matches against calibrated opponents: random, a one-ply material-greedy bot, Stockfish at its lowest Elo settings, and your own earlier checkpoints. Compute Elo with error bars.

**Checkpoint:** A rating with a confidence interval, plus a progress curve across your checkpoints.

### Day 30: Buffer and write-up

You will need the buffer.

## Elo estimate

These estimates are on an engine-style scale anchored by Stockfish's limited-strength mode. They are rough and not directly comparable to human online ratings.

For **pure self-play chess from scratch**, expect roughly 800–1,300, with a lot of uncertainty. It should reliably beat random and greedy bots and play recognisable chess, but it will likely struggle to beat Stockfish at its weakest setting (about 1,320). Early Lc0 networks trained on similar game counts were in this range.

If you **pretrain on human games first** (for example, a few million positions from the free Lichess open database), then continue with self-play, 1,500–2,000 is realistic, because supervised policy plus search is powerful even with a small network. The tradeoff is that it is no longer "zero."

For **Connect Four**, near-perfect play is achievable and you can verify it exactly against the solver. This is the most convincing demonstration that your implementation is correct.

## Practical warnings

Python MCTS will be your bottleneck, more than the GPU. Keep tree operations in plain Python or NumPy, batch everything that touches the network, and profile early. Use the M5's CPU cores for parallel self-play workers and the GPU for batched inference and training. Finally, don't skip the evaluation harness. Without reliable match statistics, you can't tell whether a change helped.
