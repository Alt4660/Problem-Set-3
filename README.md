# Econ 5200 PS3: The Rollout That Looked Like It Worked

Problem Set 3 for ECON 5200. It looks at an agency's claim that a county training program raised outcomes by 1.49 points. The estimate comes from a two-way fixed effects (TWFE) regression on a staggered rollout. I show why that number is wrong and what a correct estimate looks like.

## Files

- `Econ_5200_PS3.ipynb`: the notebook, with all four phases.
- `src/did_toolkit.py`: the module written in Step 3.3. It has the difference-in-differences functions and the clustered-inference wrapper.
- `ai-appendix.md`: the full AI transcript for Phase 4 (prompts, replies, and what I changed).
- PDF exports of the AI chats, which are the primary record behind the appendix.

## How to run

Open the notebook in Colab and use Runtime, then Restart session and run all. It runs top to bottom. The `src/did_toolkit.py` file is written by a cell in the notebook.

## What it finds

- TWFE gives 1.49, but the true average effect is 3.94. About 30% of its weight comes from comparisons that use already-treated counties as controls.
- Comparing only against not-yet-treated counties gives about 3.96.
- Clustering by board and using a wild cluster bootstrap gives a p-value of about 0.77, so the agency's result is not distinguishable from zero.

## AI use

Phase 4 used AI as required. The full exchange is in `ai-appendix.md`.
